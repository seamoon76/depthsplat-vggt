import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, N, dim]
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class PoseAdjustHead(nn.Module):
    def __init__(self, input_channels=64, dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Conv2d(input_channels, dim, kernel_size=1)  # [B, 64, 256, 256] -> [B, dim, 256, 256]
        
        self.transformer_blocks = nn.Sequential(
            *[SelfAttentionBlock(dim=dim, num_heads=num_heads) for _ in range(num_layers)]
        )

        self.output_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 9)
        )

    def forward(self, x):  # x: [B, 64, 256, 256]
        x = self.input_proj(x)                 # [B, dim, 256, 256]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)       # [B, N=256*256, dim]

        x = self.transformer_blocks(x)         # [B, N, dim]
        x = x.mean(dim=1)                      # global average pooling over N -> [B, dim]
        out = self.output_mlp(x)               # -> [B, 9]
        return out


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
