from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import pdb

@dataclass
class LossCorrCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossCorrCfgWrapper:
    corr: LossCorrCfg


class LossCorr(Loss[LossCorrCfg, LossCorrCfgWrapper]):
    corr: None 
    
    def __init__(self, cfg: LossCorrCfgWrapper) -> None:
        super().__init__(cfg)

        self.corr = None
    def huber_loss(self, pred: torch.Tensor, label: torch.Tensor, reduction: str='mean'):
        return torch.nn.functional.huber_loss(pred, label, reduction=reduction, delta=0.5) * 2.

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        valid_depth_mask: Tensor | None
    ) -> Float[Tensor, ""]:

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)
        corres_map = batch["context"]["corres_map"] # B, 3, H, W, the first 2 are corr, the last is mask for sparse match
        corres_mask = corres_map[:,2:]!=-1 # B, 1, H, W. 1 means there is valid matching pixel
        assert corres_mask.sum()!=0 # at least one valid corr 
        confidence = batch["context"]["corres_confidence"]
        context_depth = batch["context"]["context_depth"] # B,2,H,W
        context_pose = batch["context"]["extrinsics"] # B,2,4,4.   2 is for two view poses,4x4 homogenious extrinsics
        context_intrinsics = batch["context"]["intrinsics"] # B,2,3,3
        # Assume B,2,H,W shape inputs, already on same device

        B, _, H, W = corres_map.shape

        # get match point p,q from corres_map
        x_p = torch.arange(W, device=corres_map.device).view(1, 1, 1, W).expand(B, 1, H, W)  # shape B,1,H,W
        y_p = torch.arange(H, device=corres_map.device).view(1, 1, H, 1).expand(B, 1, H, W)
        p_coords = torch.cat([x_p, y_p], dim=1).float()  # shape B,2,H,W

        # choose valid match
        valid_mask = corres_mask.squeeze(1)  # B,H,W
        p_coords = p_coords.permute(0, 2, 3, 1)[valid_mask]      # [N, 2]
        q_coords = corres_map[:, :2].permute(0, 2, 3, 1)[valid_mask]  # [N, 2]

        # depth and camera
        depth_i = context_depth[:, 0:1].permute(0, 2, 3, 1)[valid_mask]  # [N, 1]
        K_i = context_intrinsics[:, 0]  # [B, 3, 3]
        K_j = context_intrinsics[:, 1]
        P_i = context_pose[:, 0]  # [B, 4, 4]
        P_j = context_pose[:, 1]

        # preprocess: make each pixel match batch index
        batch_ids = torch.arange(B, device=corres_map.device).view(B, 1, 1).expand(B, H, W)[valid_mask]

        # unpreject pixel p -> 3D point
        # unproject p: X_world = P_i^-1 * (K_i^-1 * [x, y, 1]^T * depth)
        ones = torch.ones_like(depth_i)
        p_homo = torch.cat([p_coords, ones], dim=1).unsqueeze(-1)  # [N, 3, 1]
        K_i_inv = torch.inverse(K_i[batch_ids])  # [N, 3, 3]
        cam_i_xyz = K_i_inv.bmm(p_homo).squeeze(-1) * depth_i  # [N, 3]

        # add homogeneous coord
        cam_i_xyz_h = torch.cat([cam_i_xyz, ones], dim=1).unsqueeze(-1)  # [N, 4, 1]

        # project to another image j
        P_j_mat = P_j[batch_ids]  # [N, 4, 4]
        K_j_mat = K_j[batch_ids]  # [N, 3, 3]
        proj_xyz = P_j_mat.bmm(P_i[batch_ids].inverse().bmm(cam_i_xyz_h))  # [N, 4, 1]
        proj_xyz = proj_xyz[:, :3, 0] / proj_xyz[:, 2:3, 0]  # unify divide z
        q_proj = K_j_mat.bmm(proj_xyz.unsqueeze(-1)).squeeze(-1)  # [N, 3]
        q_proj_xy = q_proj[:, :2]  # [N, 2]

        # Huber loss between projected q and ground-truth q
        loss = self.huber_loss(q_proj_xy, q_coords, reduction='none').sum(dim=-1)
        return self.cfg.weight * loss.mean()
