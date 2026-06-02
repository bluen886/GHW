import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple
from dataclasses import dataclass

import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width

        # Pre-compute pixel coordinates grid.
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing="ij",
        )

        # Shape: (H, W, 2), in (x, y) order.
        self.register_buffer("pixels", torch.stack([x, y], dim=-1))

    def compute_projection(
        self,
        means3D: torch.Tensor,  # (N, 3)
        covs3d: torch.Tensor,   # (N, 3, 3)
        K: torch.Tensor,        # (3, 3)
        R: torch.Tensor,        # (3, 3)
        t: torch.Tensor,        # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]

        # 1. Transform points to camera space.
        cam_points = means3D @ R.T + t.unsqueeze(0)  # (N, 3)

        # 2. Get depths before projection for sorting and clipping.
        raw_depths = cam_points[:, 2]                 # (N,)
        depths = raw_depths.clamp(min=1.0)            # stable denominator

        # 3. Project to screen space using camera intrinsics.
        safe_cam_points = cam_points.clone()
        safe_cam_points[:, 2] = depths
        screen_points = safe_cam_points @ K.T         # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3]  # (N, 2)

        # 4. Transform covariance to camera space and then to 2D.
        # Perspective projection:
        #   u = (k00*x + k01*y + k02*z) / z
        #   v = (k10*x + k11*y + k12*z) / z
        x = safe_cam_points[:, 0]
        y = safe_cam_points[:, 1]
        z = safe_cam_points[:, 2]

        k00, k01 = K[0, 0], K[0, 1]
        k10, k11 = K[1, 0], K[1, 1]

        J_proj = torch.zeros(
            (N, 2, 3),
            device=means3D.device,
            dtype=means3D.dtype,
        )

        J_proj[:, 0, 0] = k00 / z
        J_proj[:, 0, 1] = k01 / z
        J_proj[:, 0, 2] = -(k00 * x + k01 * y) / (z * z)

        J_proj[:, 1, 0] = k10 / z
        J_proj[:, 1, 1] = k11 / z
        J_proj[:, 1, 2] = -(k10 * x + k11 * y) / (z * z)

        # Apply world-to-camera rotation to the 3D covariance matrix:
        #   Sigma_cam = R Sigma_world R^T
        R_batch = R.to(device=means3D.device, dtype=means3D.dtype).unsqueeze(0).expand(N, -1, -1)
        covs_cam = torch.bmm(R_batch, torch.bmm(covs3d, R_batch.transpose(1, 2)))  # (N, 3, 3)

        # Project to 2D:
        #   Sigma_2D = J Sigma_cam J^T
        covs2D = torch.bmm(
            J_proj,
            torch.bmm(covs_cam, J_proj.permute(0, 2, 1))
        )  # (N, 2, 2)

        return means2D, covs2D, raw_depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,  # (N, 2)
        covs2D: torch.Tensor,   # (N, 2, 2)
        pixels: torch.Tensor,   # (H, W, 2)
    ) -> torch.Tensor:          # (N, H, W)
        N = means2D.shape[0]

        # Compute offset from mean, shape: (N, H, W, 2).
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)

        # Add small epsilon to diagonal for numerical stability.
        eps = 1e-4
        eye = torch.eye(2, device=covs2D.device, dtype=covs2D.dtype).unsqueeze(0)
        covs2D = covs2D + eps * eye

        # Manual inverse and determinant for 2x2 matrices.
        a = covs2D[:, 0, 0]
        b = covs2D[:, 0, 1]
        c = covs2D[:, 1, 0]
        d = covs2D[:, 1, 1]

        det = a * d - b * c
        det = det.clamp(min=eps)

        inv00 = d / det
        inv01 = -b / det
        inv10 = -c / det
        inv11 = a / det

        dx0 = dx[..., 0]
        dx1 = dx[..., 1]

        # Mahalanobis exponent:
        # -0.5 * (x-mu)^T Sigma^{-1} (x-mu)
        mahal = (
            dx0 * (inv00.view(N, 1, 1) * dx0 + inv01.view(N, 1, 1) * dx1)
            + dx1 * (inv10.view(N, 1, 1) * dx0 + inv11.view(N, 1, 1) * dx1)
        )
        exponent = -0.5 * mahal

        # Normalized 2D Gaussian density.
        normalizer = 2.0 * torch.pi * torch.sqrt(det).view(N, 1, 1)
        gaussian = torch.exp(exponent) / normalizer  # (N, H, W)

        return gaussian

    def forward(
        self,
        means3D: torch.Tensor,  # (N, 3)
        covs3d: torch.Tensor,   # (N, 3, 3)
        colors: torch.Tensor,   # (N, 3)
        opacities: torch.Tensor,  # (N, 1)
        K: torch.Tensor,        # (3, 3)
        R: torch.Tensor,        # (3, 3)
        t: torch.Tensor,        # (3)
    ) -> torch.Tensor:
        N = means3D.shape[0]

        # 1. Project to 2D.
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)

        # 2. Depth mask.
        valid_mask = (depths > 1.0) & (depths < 50.0)  # (N,)

        # 3. Sort by depth, front to back.
        indices = torch.argsort(depths, dim=0, descending=False)  # (N,)
        means2D = means2D[indices]       # (N, 2)
        covs2D = covs2D[indices]         # (N, 2, 2)
        colors = colors[indices]         # (N, 3)
        opacities = opacities[indices]   # (N, 1)
        valid_mask = valid_mask[indices] # (N,)

        # 4. Compute Gaussian values.
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)

        # 5. Apply valid mask.
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)

        # 6. Alpha composition setup.
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        alphas = alphas.clamp(0.0, 0.99)

        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)

        # 7. Compute alpha-blending weights.
        # T_i = prod_{j<i}(1 - alpha_j), weights_i = alpha_i * T_i.
        one_minus_alpha = 1.0 - alphas
        transmittance = torch.cumprod(
            torch.cat(
                [torch.ones_like(one_minus_alpha[:1]), one_minus_alpha + 1e-10],
                dim=0,
            ),
            dim=0,
        )[:-1]
        weights = alphas * transmittance  # (N, H, W)

        # 8. Final rendering.
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        rendered = rendered.clamp(0.0, 1.0)

        return rendered
