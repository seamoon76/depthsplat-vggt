import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from vggt.models.vggt import VGGT
from safetensors.torch import load_file
from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False
    use_index_to_load_chunk: Optional[bool] = False


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0
    

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for i, root in enumerate(cfg.roots):
            root = root / self.data_stage
            if self.cfg.use_index_to_load_chunk:
                with open(root / "index.json", "r") as f:
                    json_dict = json.load(f)
                root_chunks = sorted(list(set(json_dict.values())))
            else:
                root_chunks = sorted(
                    [path for path in root.iterdir() if path.suffix == ".torch"]
                )

            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # testing on a subset for fast speed
            self.chunks = self.chunks[::cfg.test_chunk_interval]

        self.use_vggt: bool = True
        if self.use_vggt:
            self.vggt_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.vggt_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            # init vggt
            self.vggt_model = VGGT()
            local_model_path = "/work/courses/3dv/35/vggt/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9/model.safetensors"
            state_dict = load_file(local_model_path)
            self.vggt_model.load_state_dict(state_dict)
            self.vggt_model.eval()
            self.vggt_model = self.vggt_model.to(self.vggt_device)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            times_per_scene = (
                1
                if self.stage == "test"
                else self.cfg.train_times_per_scene
            )

            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                if self.use_vggt:
                    center_cropped_context_images_vggt, h_cropped, w_cropped = self.vggt_convert_images(context_images)
                context_images = self.convert_images(context_images)


                if self.use_vggt:
                    extrinsics_vggt, intrinsics_vggt = self.estimate_camera_vggt(center_cropped_context_images_vggt, h_cropped, w_cropped)
                    extrinsics_vggt.to(extrinsics.device)
                    intrinsics_vggt.to(intrinsics.device)

                print(f"H={h_cropped},W={w_cropped}")
                
                
                
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)

                # Skip the example if the images don't have the right shape.
                if self.cfg.highres:
                    expected_shape = (3, 720, 1280)
                else:
                    expected_shape = (3, 360, 640)
                context_image_invalid = context_images.shape[1:] != expected_shape
                target_image_invalid = target_images.shape[1:] != expected_shape
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                nf_scale = 1.0
                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices] if not use_vggt else extrinsics_vggt[context_indices],
                        "intrinsics": intrinsics[context_indices] if not use_vggt else intrinsics_vggt[context_indices],
                        "image": context_images,
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices] if not use_vggt else extrinsics_vggt[context_indices],
                        "intrinsics": intrinsics[target_indices] if not use_vggt else intrinsics_vggt[context_indices],
                        "image": target_images,
                        "near": self.get_bound("near", len(target_indices)) / nf_scale,
                        "far": self.get_bound("far", len(target_indices)) / nf_scale,
                        "index": target_indices,
                    },
                    "scene": scene,
                }

                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for i, root in enumerate(self.cfg.roots):
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()), self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.train_times_per_scene
        )
    
    def vggt_convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
        patch_size = 14
    ):
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        images_tensor = torch.stack(torch_images)
        _, _, h, w = images_tensor.shape

        # Image size must be even so that naive center-cropping does not cause misalignment.
        assert h % 2 == 0 and w % 2 == 0

        h_new = (h // patch_size) * patch_size
        row = (h - h_new) // 2
        w_new = (w // patch_size) * patch_size
        col = (w - w_new) // 2

        # Center-crop the image.
        image = images_tensor[:, :, row: row + h_new, col: col + w_new]
        h_cropped = image.shape[-2]
        w_cropped = image.shape[-1]
        return image, h_cropped, w_cropped
    
    def estimate_camera_vggt(self, images, H, W):
        """
        inputs:
        images: tensor [N,3,H,W] patched with 14
        
        return:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): Camera extrinsic parameters with shape Sx3x4.
              In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world
              transformation. The format is [R|t] where R is a 3x3 rotation matrix and t is
              a 3x1 translation vector.
            - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape Sx3x3,
              or None if build_intrinsics is False. Defined in pixels, with format:
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              where fx, fy are focal lengths and (cx, cy) is the principal point,
              assumed to be at the center of the image (W/2, H/2).
              then normalized
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.vggt_dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                # predictions = model(images)
                images = images[None].to(self.vggt_device)  # add batch dimension
                aggregated_tokens_list, ps_idx = self.vggt_model.aggregator(images)
            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            extrinsic = extrinsic[0] # remove batch dim
            intrinsic = intrinsic[0]
            W=W.to(self.vggt_device)
            H=H.to(self.vggt_device)
            intrinsic_clone = intrinsic.clone()
            intrinsic_clone[..., 0, :] /= W
            intrinsic_clone[..., 1, :] /= H
            
            return extrinsic, intrinsic
        
        
