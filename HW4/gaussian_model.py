import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass

# pytorch3d is optional: used only for a KNN distance query at init. If it is
# not importable we fall back to a pure-PyTorch KNN returning the SAME thing
# (squared distances, self included), so nothing else changes and no CUDA
# compilation is required.
try:
    from pytorch3d.ops.knn import knn_points  # noqa: F401
    _HAS_PYTORCH3D = True
except Exception:
    _HAS_PYTORCH3D = False


def _knn_sq_dists_torch(points: torch.Tensor, K: int, chunk: int = 4096) -> torch.Tensor:
    """Pure-PyTorch replacement for pytorch3d.knn_points (distances only).
    points: (1, N, 3) -> returns (1, N, K) SQUARED distances."""
    pts = points[0]
    N = pts.shape[0]
    out = torch.empty((N, K), device=pts.device, dtype=pts.dtype)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d = torch.cdist(pts[s:e], pts)
        vals, _ = torch.topk(d, k=K, dim=1, largest=False)
        out[s:e] = vals * vals
    return out.unsqueeze(0)


@dataclass
class GaussianParameters:
    positions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    covariance: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor


class GaussianModel(nn.Module):
    def __init__(self, points3D_xyz: torch.Tensor, points3D_rgb: torch.Tensor):
        super().__init__()
        self.n_points = len(points3D_xyz)
        self._init_positions(points3D_xyz)
        self._init_rotations()
        self._init_scales(points3D_xyz)
        self._init_colors(points3D_rgb)
        self._init_opacities()

    def _init_positions(self, points3D_xyz: torch.Tensor) -> None:
        self.positions = nn.Parameter(torch.as_tensor(points3D_xyz, dtype=torch.float32))

    def _init_rotations(self) -> None:
        initial_rotations = torch.zeros((self.n_points, 4))
        initial_rotations[:, 0] = 1.0
        self.rotations = nn.Parameter(initial_rotations)

    def _init_scales(self, points3D_xyz: torch.Tensor) -> None:
        K = min(50, self.n_points - 1)
        points = torch.as_tensor(points3D_xyz, dtype=torch.float32).unsqueeze(0)
        if _HAS_PYTORCH3D:
            dists, _, _ = knn_points(points, points, K=K)
        else:
            dists = _knn_sq_dists_torch(points, K=K)
        mean_dists = torch.mean(torch.sqrt(dists[0]), dim=1, keepdim=True) * 2.
        mean_dists = mean_dists.clamp(0.2 * torch.median(mean_dists),
                                      3.0 * torch.median(mean_dists))
        print('init_scales', torch.min(mean_dists), torch.max(mean_dists))
        log_scales = torch.log(mean_dists)
        self.scales = nn.Parameter(log_scales.repeat(1, 3))

    def _init_colors(self, points3D_rgb: torch.Tensor) -> None:
        colors = torch.as_tensor(points3D_rgb, dtype=torch.float32) / 255.0
        colors = colors.clamp(0.001, 0.999)
        self.colors = nn.Parameter(torch.logit(colors))

    def _init_opacities(self) -> None:
        self.opacities = nn.Parameter(8.0 * torch.ones((self.n_points, 1), dtype=torch.float32))

    def _compute_rotation_matrices(self) -> torch.Tensor:
        q = F.normalize(self.rotations, dim=-1)
        w, x, y, z = q.unbind(-1)
        R00 = 1 - 2*y*y - 2*z*z
        R01 = 2*x*y - 2*w*z
        R02 = 2*x*z + 2*w*y
        R10 = 2*x*y + 2*w*z
        R11 = 1 - 2*x*x - 2*z*z
        R12 = 2*y*z - 2*w*x
        R20 = 2*x*z - 2*w*y
        R21 = 2*y*z + 2*w*x
        R22 = 1 - 2*x*x - 2*y*y
        return torch.stack([R00, R01, R02, R10, R11, R12, R20, R21, R22], dim=-1).reshape(-1, 3, 3)

    def compute_covariance(self) -> torch.Tensor:
        R = self._compute_rotation_matrices()
        scales = torch.exp(self.scales)
        S = torch.diag_embed(scales)
        # Sigma = R S S^T R^T = (R S)(R S)^T   (paper Eq. 6)
        M = torch.bmm(R, S)
        Covs3d = torch.bmm(M, M.transpose(1, 2))
        return Covs3d

    def get_gaussian_params(self) -> GaussianParameters:
        return GaussianParameters(
            positions=self.positions,
            colors=torch.sigmoid(self.colors),
            opacities=torch.sigmoid(self.opacities),
            covariance=self.compute_covariance(),
            rotations=F.normalize(self.rotations, dim=-1),
            scales=torch.clamp(torch.exp(self.scales), max=0.5)
        )

    def forward(self) -> Dict[str, torch.Tensor]:
        params = self.get_gaussian_params()
        return {
            'positions': params.positions,
            'covariance': params.covariance,
            'colors': params.colors,
            'opacities': params.opacities
        }
