import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
from typing import Tuple
from config import DEVICE, DTYPE, OBJECT_NAMES, logger
from utils import create_meshgrid

class LieGroupRepresentation(nn.Module):
    def __init__(self, group_type: str = "SE(2)"):
        super().__init__()
        self.group_type: str = group_type
        self.param_dim: int = 3 if group_type == "SE(2)" else 6 if group_type == "SE(3)" else 7

    def lie_algebra_to_params(self, alg: torch.Tensor) -> torch.Tensor:
        """Convert Lie algebra to transformation parameters."""
        if self.group_type == "SE(2)":
            theta, tx, ty = alg[..., 0], alg[..., 1], alg[..., 2]
            return torch.stack([theta, tx, ty], dim=-1)
        elif self.group_type == "SE(3)":
            rot = alg[..., :3]
            trans = alg[..., 3:]
            return torch.cat([rot, trans], dim=-1)
        elif self.group_type == "Sim(3)":
            rot = alg[..., :3]
            trans = alg[..., 3:6]
            scale = alg[..., 6]
            return torch.cat([rot, trans, scale.unsqueeze(-1)], dim=-1)
        raise ValueError(f"Unsupported group: {self.group_type}")

    def params_to_matrix(self, params: torch.Tensor) -> torch.Tensor:
        """Convert parameters to transformation matrix."""
        if self.group_type == "SE(2)":
            theta, tx, ty = params[..., 0], params[..., 1], params[..., 2]
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            matrix = torch.stack([
                cos_t, -sin_t, tx,
                sin_t, cos_t, ty,
                torch.zeros_like(tx), torch.zeros_like(tx), torch.ones_like(tx)
            ], dim=-1).view(*params.shape[:-1], 3, 3)
            return matrix
        elif self.group_type == "SE(3)":
            rot = K.geometry.so3.exp_to_rot(params[..., :3])
            trans = params[..., 3:]
            matrix = torch.eye(4, device=params.device, dtype=params.dtype).repeat(*params.shape[:-1], 1, 1)
            matrix[..., :3, :3] = rot
            matrix[..., :3, 3] = trans
            return matrix
        elif self.group_type == "Sim(3)":
            rot = K.geometry.so3.exp_to_rot(params[..., :3])
            trans = params[..., 3:6]
            scale = params[..., 6]
            matrix = torch.eye(4, device=params.device, dtype=params.dtype).repeat(*params.shape[:-1], 1, 1)
            matrix[..., :3, :3] = scale.unsqueeze(-1).unsqueeze(-1) * rot
            matrix[..., :3, 3] = trans
            return matrix
        raise ValueError(f"Unsupported group: {self.group_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lie_algebra_to_params(x)

class LieGroupConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, group_type: str = "SE(2)"):
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2).to(DEVICE)
        self.lie_rep: LieGroupRepresentation = LieGroupRepresentation(group_type)
        self.group_samples: int = 4
        self.group_type: str = group_type

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = x.to(DEVICE)
        g = g.to(DEVICE)
        outputs = []
        for i in range(self.group_samples):
            g_sample = g * (i / self.group_samples)
            transformed_x = self.apply_group_action(x, g_sample)
            out = self.conv(transformed_x)
            outputs.append(out)
        return torch.mean(torch.stack(outputs), dim=0)

    def apply_group_action(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        matrix = self.lie_rep.params_to_matrix(g)
        if self.group_type in ["SE(3)", "Sim(3)"] and x.dim() == 4:
            logger.warning("3D group applied to 2D image, using 2D projection")
            matrix = matrix[..., :3, :3]
        grid = create_meshgrid(x.shape[2], x.shape[3], x.device)
        warped_grid = K.geometry.transform_points(matrix.unsqueeze(0), grid.unsqueeze(0))
        return F.grid_sample(x, warped_grid, align_corners=True)

class LieGroupAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, group_type: str = "SE(2)"):
        super().__init__()
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(embed_dim, num_heads).to(DEVICE)
        self.lie_rep: LieGroupRepresentation = LieGroupRepresentation(group_type)
        self.embed_dim: int = embed_dim
        self.group_projection: nn.Linear = nn.Linear(self.lie_rep.param_dim, embed_dim).to(DEVICE)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        q, k, v, g = q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), g.to(DEVICE)
        rel_g = self.compute_relative_g(g)
        rel_g = self.group_projection(rel_g)
        q = q + rel_g
        k = k + rel_g
        attn_output, _ = self.attn(q, k, v)
        return attn_output

    def compute_relative_g(self, g: torch.Tensor) -> torch.Tensor:
        return g.unsqueeze(1) - g.unsqueeze(0)

class TransformerModel(nn.Module):
    def __init__(self, num_classes: int = len(OBJECT_NAMES), embed_dim: int = 64, num_heads: int = 2, num_layers: int = 2, group_type: str = "SE(2)"):
        super().__init__()
        self.embed_dim: int = embed_dim
        self.lie_conv: LieGroupConv = LieGroupConv(3, embed_dim, kernel_size=3, group_type=group_type)
        self.lie_attn_layers: nn.ModuleList = nn.ModuleList([LieGroupAttention(embed_dim, num_heads, group_type=group_type) for _ in range(num_layers)])
        self.fc: nn.Linear = nn.Linear(embed_dim, num_classes).to(DEVICE)
        self.norm: nn.LayerNorm = nn.LayerNorm(embed_dim).to(DEVICE)
        self.lie_rep: LieGroupRepresentation = LieGroupRepresentation(group_type=group_type)

    def forward(self, x: torch.Tensor, lie_params: torch.Tensor) -> torch.Tensor:
        x = self.lie_conv(x, lie_params)
        x = x.flatten(2).permute(2, 0, 1)
        for attn in self.lie_attn_layers:
            res = x
            x = attn(x, x, x, lie_params)
            x = self.norm(x + res)
        x = x.mean(0)
        return self.fc(x.unsqueeze(0))

class GeometricLoss(nn.Module):
    def __init__(self, group_samples: int = 4):
        super().__init__()
        self.group_samples: int = group_samples
        self.ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.inv_loss_weight: float = 0.5

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, model: nn.Module, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        outputs, labels, x, g = outputs.to(DEVICE), labels.to(DEVICE), x.to(DEVICE), g.to(DEVICE)
        ce = self.ce_loss(outputs, labels)
        inv_loss = 0
        for i in range(self.group_samples):
            g_sample = g * (i / self.group_samples)
            transformed_x = model.lie_conv.apply_group_action(x, g_sample)
            transformed_out = model(transformed_x, g_sample)
            inv_loss += F.mse_loss(outputs, transformed_out)
        return ce + self.inv_loss_weight * (inv_loss / self.group_samples)