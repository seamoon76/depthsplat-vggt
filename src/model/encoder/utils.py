import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseAdjustHead(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=128):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> [B, C, 1, 1]
        self.fc = nn.Sequential(
            nn.Flatten(),               # -> [B, C]
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9)    # output [B, 9]
        )

    def forward(self, feat):  # feat: [B, C, H, W]
        x = self.global_pool(feat)
        out = self.fc(x)
        return out  # [B, 9]

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
