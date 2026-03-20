"""
Thin Plate Spline Spatial Transformer Network (TPS-STN)
Nắn thẳng biển số bị cong/biến dạng
Thay thế Affine transformer để xử lý:
- Motorcycle plates (curved edges)
- Fisheye distortion
- Perspective angle
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class TPSGridGenerator(nn.Module):
    """
    Generate TPS transformation grid
    Input: Control points (batch_size, num_points, 2)
    Output: Transformation grid for grid_sample
    """

    def __init__(self, out_h: int = 32, out_w: int = 64, num_points: int = 16):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.num_points = num_points

        grid_size = int(math.sqrt(num_points))
        assert grid_size * grid_size == num_points, f"num_points must be perfect square, got {num_points}"

        ctrl_pts = torch.linspace(-1, 1, grid_size)
        grid = torch.meshgrid(ctrl_pts, ctrl_pts, indexing='ij')
        self.register_buffer('ctrl_grid', torch.stack([grid[0].flatten(), grid[1].flatten()], dim=1))

    @staticmethod
    def _build_Lp(cp: torch.Tensor) -> torch.Tensor:
        N = cp.shape[0]

        pairwise_diff = cp.unsqueeze(0) - cp.unsqueeze(1)
        pairwise_dist = torch.sqrt(torch.sum(pairwise_diff**2, dim=2))

        Lp = pairwise_dist**2 * torch.log(pairwise_dist + 1e-8)

        return Lp

    @staticmethod
    def _build_P(cp: torch.Tensor) -> torch.Tensor:
        N = cp.shape[0]
        one = torch.ones(N, 1, device=cp.device)
        P = torch.cat([one, cp], dim=1)

        return P

    def compute_weights(self, source_pts: torch.Tensor, target_pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = source_pts.shape[0]
        N = self.num_points

        Lp = self._build_Lp(source_pts[0])
        P = self._build_P(source_pts[0])

        M = torch.zeros(N + 3, N + 3, device=source_pts.device)
        M[:N, :N] = Lp
        M[:N, N:] = P
        M[N:, :N] = P.t()

        M_inv = torch.linalg.pinv(M)  # pinv is safe for near-singular matrices

        weights = []
        affine_params = []

        for b in range(batch_size):
            b_target = torch.cat([target_pts[b], torch.zeros(3, 2, device=target_pts.device)], dim=0)

            coeffs = torch.mm(M_inv, b_target)

            w = coeffs[:N]
            a = coeffs[N:]

            weights.append(w)
            affine_params.append(a)

        return torch.stack(weights), torch.stack(affine_params)

    def forward(self, control_points_offset: torch.Tensor) -> torch.Tensor:
        batch_size = control_points_offset.shape[0]

        source_pts = self.ctrl_grid.unsqueeze(0).expand(batch_size, -1, -1)
        target_pts = source_pts + control_points_offset

        w, a = self.compute_weights(source_pts, target_pts)

        grid_y = torch.linspace(-1, 1, self.out_h, device=control_points_offset.device)
        grid_x = torch.linspace(-1, 1, self.out_w, device=control_points_offset.device)
        grid_xx, grid_yy = torch.meshgrid(grid_x, grid_y, indexing='xy')
        query_pts = torch.stack([grid_xx, grid_yy], dim=-1)

        query_pts = query_pts.unsqueeze(0).expand(batch_size, -1, -1, -1)

        grids = []

        for b in range(batch_size):
            pairwise_diff = query_pts[b].unsqueeze(2) - source_pts[b].unsqueeze(0).unsqueeze(0)
            pairwise_dist = torch.sqrt(torch.sum(pairwise_diff**2, dim=-1) + 1e-8)

            K = pairwise_dist**2 * torch.log(pairwise_dist + 1e-8)

            transformed = torch.einsum('hwn,nd->hwd', K, w[b])

            one = torch.ones_like(query_pts[b, :, :, :1])
            P = torch.cat([one, query_pts[b]], dim=-1)
            transformed = transformed + torch.einsum('hwd,nd->hwn', P, a[b]).unsqueeze(-1).squeeze(-1)

            grids.append(transformed)

        return torch.stack(grids)


class TPSSTN(nn.Module):
    """
    Thin Plate Spline Spatial Transformer Network
    Takes plate image and learns optimal control point offsets
    """

    def __init__(self, in_channels: int = 3, num_fiducial: int = 16,
                 tps_grid_size: int = 4, tps_output_size: Tuple[int, int] = (32, 100)):
        super().__init__()
        self.in_channels = in_channels
        self.num_fiducial = num_fiducial
        self.tps_grid_size = tps_grid_size
        self.tps_output_size = tps_output_size

        # Guard: FC outputs num_fiducial offsets but TPS grid uses tps_grid_size²
        num_points = tps_grid_size * tps_grid_size
        assert num_fiducial == num_points, (
            f"num_fiducial ({num_fiducial}) must equal tps_grid_size² "
            f"({tps_grid_size}² = {num_points})"
        )

        self.localization_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_fiducial * 2)
        )

        self.tps_generator = TPSGridGenerator(
            out_h=tps_output_size[0],
            out_w=tps_output_size[1],
            num_points=num_points
        )

        self._init_fc()

    def _init_fc(self):
        with torch.no_grad():
            self.fc[-1].weight.zero_()
            self.fc[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        feat = self.localization_net(x)
        feat = feat.view(batch_size, -1)

        offsets = self.fc(feat)
        offsets = offsets.view(batch_size, self.num_fiducial, 2)

        grid = self.tps_generator(offsets)

        warped = F.grid_sample(
            x,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped, grid
