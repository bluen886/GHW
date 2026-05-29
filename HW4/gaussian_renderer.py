import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))

    def compute_projection(self, means3D, covs3d, K, R, t):
        N = means3D.shape[0]
        cam_points = means3D @ R.T + t.unsqueeze(0)
        depths = cam_points[:, 2].clamp(min=0.1) # 严格防止除零
        screen_points = cam_points @ K.T
        means2D = screen_points[..., :2] / screen_points[..., 2:3]

        fx = K[0, 0]
        fy = K[1, 1]
        X = cam_points[:, 0]
        Y = cam_points[:, 1]
        Z = depths
        inv_Z = 1.0 / Z
        inv_Z2 = inv_Z * inv_Z
        
        J_proj = torch.zeros((N, 2, 3), device=means3D.device, dtype=means3D.dtype)
        J_proj[:, 0, 0] = fx * inv_Z
        J_proj[:, 0, 2] = -fx * X * inv_Z2
        J_proj[:, 1, 1] = fy * inv_Z
        J_proj[:, 1, 2] = -fy * Y * inv_Z2

        covs_cam = R @ covs3d @ R.T
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))
        
        # 【修改点 1】为 2D 协方差矩阵引入低通膨胀滤波（防止高斯过小或非正定），并严格截断过大高斯
        # 相当于在屏幕空间加上一个像素大小的微小高斯核
        covs2D[:, 0, 0] = covs2D[:, 0, 0] + 0.3
        covs2D[:, 1, 1] = covs2D[:, 1, 1] + 0.3
        return means2D, covs2D, depths

    def compute_gaussian_values(self, means2D, covs2D, pixels):
        N = means2D.shape[0]
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        a = covs2D[:, 0, 0]
        b = covs2D[:, 0, 1]
        c = covs2D[:, 1, 0]
        d = covs2D[:, 1, 1]
        
        det = a * d - b * c
        # 【修改点 2】严格保证行列式大于给定正数，防止奇异矩阵求逆带来的梯度爆炸
        det = torch.clamp(det, min=1e-3)
        
        inv_cov = torch.zeros_like(covs2D)
        inv_cov[:, 0, 0] = d / det
        inv_cov[:, 0, 1] = -b / det
        inv_cov[:, 1, 0] = -c / det
        inv_cov[:, 1, 1] = a / det

        # 【修改点 3】限制各向异性惩罚的最大范围，防止某些过长的高斯点主导整张图
        dx_inv = torch.einsum('nhwi,nij->nhwj', dx, inv_cov)
        P = -0.5 * (dx_inv * dx).sum(dim=-1)
        P = torch.clamp(P, min=-20.0, max=0.0) # 防止 exp(P) 溢出或下溢
        gaussian = torch.exp(P)
        return gaussian

    def forward(self, means3D, covs3d, colors, opacities, K, R, t):
        N = means3D.shape[0]
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 仅保留合理的视锥内的点
        valid_mask = (depths > 0.1) & (depths < 50.0)
        
        # 按深度从小到大（由前到后）排序
        indices = torch.argsort(depths, dim=0, descending=False)
        means2D = means2D[indices]
        covs2D = covs2D[indices]
        colors = colors[indices]
        opacities = opacities[indices]
        valid_mask = valid_mask[indices]

        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)

        alphas = opacities.view(N, 1, 1) * gaussian_values
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)
        colors = colors.permute(0, 2, 3, 1)

        # 【修改点 4】前向混合的数值平滑截断，防止单点吃掉所有透射率
        alphas = alphas.clamp(0.0, 0.99)
        T = torch.cumprod(1.0 - alphas, dim=0)
        T = torch.cat([torch.ones_like(T[:1]), T[:-1]], dim=0)
        weights = alphas * T

        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)
        return rendered
