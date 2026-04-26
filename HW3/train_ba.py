#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 3 - Bundle Adjustment with PyTorch

This script optimizes:
1. shared focal length f
2. 50 camera extrinsics R, T
3. 20000 3D point coordinates

Input data:
    data/points2d.npz
    data/points3d_colors.npy

Run:
    python train_ba.py --data_dir data --iters 3000 --batch_size 262144 --device cuda

Main outputs:
    outputs/loss_curve.png
    outputs/reconstruction.obj
    outputs/optimized_points.npy
    outputs/cameras.npz
    outputs/metrics.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to rotation matrices without requiring PyTorch3D."""
    x, y, z = euler.unbind(-1)

    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], -1),
            torch.stack([zeros, cx, -sx], -1),
            torch.stack([zeros, sx, cx], -1),
        ],
        -2,
    )
    ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], -1),
            torch.stack([zeros, ones, zeros], -1),
            torch.stack([-sy, zeros, cy], -1),
        ],
        -2,
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, zeros], -1),
            torch.stack([sz, cz, zeros], -1),
            torch.stack([zeros, zeros, ones], -1),
        ],
        -2,
    )

    # Same convention as R = Rz @ Ry @ Rx.
    return rz @ ry @ rx


class BundleAdjustmentModel(nn.Module):
    def __init__(
        self,
        init_points: np.ndarray,
        n_views: int,
        image_size: int = 1024,
        init_focal: float = 900.0,
        init_dist: float = 2.5,
        yaw_range_deg: float = 70.0,
    ) -> None:
        super().__init__()

        self.image_size = float(image_size)
        self.cx = float(image_size) / 2.0
        self.cy = float(image_size) / 2.0

        self.points = nn.Parameter(torch.tensor(init_points, dtype=torch.float32))

        # The assignment states views lie around the front side with about +/- 70 degrees.
        euler = torch.zeros(n_views, 3, dtype=torch.float32)
        if n_views > 1:
            euler[:, 1] = torch.linspace(
                math.radians(-yaw_range_deg), math.radians(yaw_range_deg), n_views
            )
        self.euler = nn.Parameter(euler)

        # Camera transform: Xc = R @ X + T. Object is on negative camera Z if T_z < 0.
        trans = torch.zeros(n_views, 3, dtype=torch.float32)
        trans[:, 2] = -float(init_dist)
        self.trans = nn.Parameter(trans)

        # Positive focal length by construction.
        self.log_focal = nn.Parameter(torch.tensor(math.log(init_focal), dtype=torch.float32))

    @property
    def focal(self) -> torch.Tensor:
        return torch.exp(self.log_focal)

    def project(self, view_ids: torch.Tensor, point_ids: torch.Tensor) -> torch.Tensor:
        pts = self.points[point_ids]                      # (B, 3)
        rot = euler_xyz_to_matrix(self.euler[view_ids])   # (B, 3, 3)
        trans = self.trans[view_ids]                      # (B, 3)

        cam = torch.bmm(rot, pts[..., None]).squeeze(-1) + trans
        x, y, z = cam.unbind(-1)

        eps = 1e-6
        z = torch.where(z.abs() < eps, torch.full_like(z, -eps), z)

        u = -self.focal * x / z + self.cx
        v =  self.focal * y / z + self.cy
        return torch.stack([u, v], dim=-1)


def load_data(data_dir: Path):
    obs_file = data_dir / "points2d.npz"
    if not obs_file.exists():
        raise FileNotFoundError(f"Cannot find {obs_file}. Run this script in the assignment folder.")

    points2d = np.load(obs_file)
    keys = sorted(points2d.files)

    all_obs = []
    view_ids, point_ids, xy = [], [], []
    for view_index, key in enumerate(keys):
        arr = points2d[key].astype(np.float32)
        all_obs.append(arr)
        visible = arr[:, 2] > 0.5
        ids = np.where(visible)[0]

        view_ids.append(np.full(ids.shape[0], view_index, dtype=np.int64))
        point_ids.append(ids.astype(np.int64))
        xy.append(arr[ids, :2].astype(np.float32))

    return (
        keys,
        all_obs,
        np.concatenate(view_ids),
        np.concatenate(point_ids),
        np.concatenate(xy),
    )


def initialize_points(all_obs, image_size: int, init_focal: float, init_dist: float) -> np.ndarray:
    """Back-project middle-view observations to obtain a stable initialization."""
    n_views = len(all_obs)
    n_points = all_obs[0].shape[0]
    ref = all_obs[n_views // 2]
    xy = ref[:, :2].copy()
    ref_visible = ref[:, 2] > 0.5

    # For points invisible in the middle view, use the average of all visible observations.
    if not ref_visible.all():
        sum_xy = np.zeros((n_points, 2), dtype=np.float32)
        cnt = np.zeros((n_points,), dtype=np.float32)
        for obs in all_obs:
            visible = obs[:, 2] > 0.5
            sum_xy[visible] += obs[visible, :2]
            cnt[visible] += 1.0

        valid = cnt > 0
        xy[~ref_visible & valid] = sum_xy[~ref_visible & valid] / cnt[~ref_visible & valid, None]
        xy[~valid] = image_size / 2.0

    cx = cy = image_size / 2.0
    # Approximate inverse projection when camera is initialized at z = -init_dist.
    X = (xy[:, 0] - cx) * init_dist / init_focal
    Y = -(xy[:, 1] - cy) * init_dist / init_focal
    Z = np.zeros_like(X)

    points = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    points += np.random.normal(0.0, 0.01, points.shape).astype(np.float32)
    return points


def write_obj(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    colors = colors.astype(np.float32)
    if colors.max() > 1.0:
        colors /= 255.0

    with open(path, "w", encoding="utf-8") as f:
        for point, color in zip(points, colors):
            f.write(
                "v {:.7f} {:.7f} {:.7f} {:.6f} {:.6f} {:.6f}\n".format(
                    point[0], point[1], point[2], color[0], color[1], color[2]
                )
            )


@torch.no_grad()
def evaluate_full_rmse(model, view_ids, point_ids, xy, chunk: int = 262144) -> float:
    errors = []
    for start in range(0, xy.shape[0], chunk):
        end = min(start + chunk, xy.shape[0])
        pred = model.project(view_ids[start:end], point_ids[start:end])
        err2 = ((pred - xy[start:end]) ** 2).sum(dim=-1)
        errors.append(err2.detach().cpu())
    errors = torch.cat(errors)
    return float(torch.sqrt(errors.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--out_dir", default="outputs", type=str)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--iters", default=3000, type=int)
    parser.add_argument("--batch_size", default=262144, type=int)
    parser.add_argument("--init_focal", default=900.0, type=float)
    parser.add_argument("--init_dist", default=2.5, type=float)
    parser.add_argument("--lr_points", default=1e-2, type=float)
    parser.add_argument("--lr_camera", default=2e-3, type=float)
    parser.add_argument("--lr_focal", default=5e-4, type=float)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--eval_every", default=100, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys, all_obs, view_ids_np, point_ids_np, xy_np = load_data(data_dir)
    n_views = len(keys)
    n_points = all_obs[0].shape[0]
    n_obs = xy_np.shape[0]

    print(f"Loaded {n_views} views, {n_points} points, {n_obs} visible observations.")

    init_points = initialize_points(all_obs, args.image_size, args.init_focal, args.init_dist)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = BundleAdjustmentModel(
        init_points=init_points,
        n_views=n_views,
        image_size=args.image_size,
        init_focal=args.init_focal,
        init_dist=args.init_dist,
    ).to(device)

    view_ids = torch.tensor(view_ids_np, dtype=torch.long, device=device)
    point_ids = torch.tensor(point_ids_np, dtype=torch.long, device=device)
    xy = torch.tensor(xy_np, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(
        [
            {"params": [model.points], "lr": args.lr_points},
            {"params": [model.euler, model.trans], "lr": args.lr_camera},
            {"params": [model.log_focal], "lr": args.lr_focal},
        ]
    )

    losses = []
    rmse_history = []

    for it in range(1, args.iters + 1):
        if 0 < args.batch_size < n_obs:
            batch = torch.randint(0, n_obs, (args.batch_size,), device=device)
        else:
            batch = torch.arange(n_obs, device=device)

        pred = model.project(view_ids[batch], point_ids[batch])
        target = xy[batch]

        # Smooth L1 is robust during early optimization.
        reproj_loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=10.0)

        # Small regularization suppresses degenerate depth drift without dominating reprojection.
        depth_reg = 1e-5 * torch.mean(model.points[:, 2] ** 2)
        trans_reg = 1e-6 * torch.mean(model.trans[:, :2] ** 2)
        loss = reproj_loss + depth_reg + trans_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

        if it == 1 or it % args.eval_every == 0 or it == args.iters:
            rmse = evaluate_full_rmse(model, view_ids, point_ids, xy)
            rmse_history.append({"iter": it, "rmse_px": rmse, "loss": losses[-1]})
            print(
                f"iter {it:05d} | loss {losses[-1]:.6f} | full RMSE {rmse:.4f}px | focal {float(model.focal):.3f}"
            )

    with torch.no_grad():
        final_points = model.points.detach().cpu().numpy()
        final_euler = model.euler.detach().cpu().numpy()
        final_trans = model.trans.detach().cpu().numpy()
        final_focal = float(model.focal.detach().cpu())

    colors_file = data_dir / "points3d_colors.npy"
    if colors_file.exists():
        colors = np.load(colors_file)
    else:
        colors = np.ones((n_points, 3), dtype=np.float32)

    np.save(out_dir / "optimized_points.npy", final_points)
    np.savez(
        out_dir / "cameras.npz",
        view_keys=np.array(keys),
        euler=final_euler,
        trans=final_trans,
        focal=final_focal,
    )
    write_obj(out_dir / "reconstruction.obj", final_points, colors)

    plt.figure(figsize=(8, 4.8))
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Smooth L1 reprojection loss")
    plt.title("Bundle Adjustment Optimization")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=180)
    plt.close()

    metrics = {
        "n_views": n_views,
        "n_points": n_points,
        "n_visible_observations": int(n_obs),
        "iterations": args.iters,
        "final_loss": losses[-1],
        "final_rmse_px": rmse_history[-1]["rmse_px"],
        "final_focal": final_focal,
        "rmse_history": rmse_history,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nFinished.")
    print(f"Final focal length: {final_focal:.4f}")
    print(f"Final full RMSE: {metrics['final_rmse_px']:.4f}px")
    print(f"Saved OBJ: {out_dir / 'reconstruction.obj'}")
    print(f"Saved loss curve: {out_dir / 'loss_curve.png'}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
