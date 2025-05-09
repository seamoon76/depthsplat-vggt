import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional
import pdb
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
import pdb
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

        self.debug=False
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
            if self.debug:
                root_chunks = [root_chunks[0]]

            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # testing on a subset for fast speed
            self.chunks = self.chunks[::cfg.test_chunk_interval]

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
            chunk = torch.load(chunk_path,map_location='cpu')
            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)
            if self.debug:
                item = [x for x in chunk if x["key"] == "5aca87f95a9412c6"]
                assert len(item)==1
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
                extrinsics_vggt_finetune, intrinsics_vggt_finetune = self.convert_poses(example["vggt_camera"])
                far_near_vggt_finetune = example["depth"]
                scene = example["key"]

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                    if self.debug:
                        print(context_indices)
                        context_indices = torch.tensor([58,70])
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
                context_images = self.convert_images(context_images)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)
                # load the far and near
                if "depth" in example.keys():
                    context_fars = torch.tensor([example["depth"][index.item()][0] for index in context_indices])
                    context_nears = torch.tensor([example["depth"][index.item()][1] for index in context_indices])
                    target_fars = torch.tensor([example["depth"][index.item()][0] for index in target_indices])
                    target_nears = torch.tensor([example["depth"][index.item()][1] for index in target_indices])

                else:
                    context_fars = self.get_bound("far",len(context_indices))
                    context_nears = self.get_bound("near",len(context_indices))
                    target_fars = self.get_bound("far",len(target_indices))
                    target_nears = self.get_bound("near",len(target_indices))
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
                
                camera_norm_matrix = extrinsics[context_indices[0]].unsqueeze(0).inverse()
                extrinsics = torch.bmm(camera_norm_matrix.repeat(extrinsics.shape[0], 1, 1), extrinsics)
                camera_norm_matrix_vggt_finetune = extrinsics_vggt_finetune[context_indices[0]].unsqueeze(0).inverse()
                extrinsics_vggt_finetune = torch.bmm(camera_norm_matrix_vggt_finetune.repeat(extrinsics_vggt_finetune.shape[0], 1, 1), extrinsics_vggt_finetune)
                nf_scale = 1.0
                extrinsics_vggt_5a = torch.tensor([[[1.000000, -0.000056, -0.000059, -0.000007],
[0.000056, 1.000000, -0.000027, -0.000018],
[0.000059, 0.000027, 1.000000, 0.000016],
[0.0, 0.0, 0.0, 1.0]],

[[0.999871, 0.004153, 0.015515, -0.010875],
[-0.004165, 0.999991, 0.000711, 0.003780],
[-0.015511, -0.000776, 0.999879, -0.061710],
[0.0, 0.0, 0.0, 1.0]]])
                extrinsics_vggt_gt_translation = torch.tensor([[[1.000000, -0.000056, -0.000059, 0.1020],
[0.000056, 1.000000, -0.000027, 0.0571],
[0.000059, 0.000027, 1.000000, -1.2647],
[0.0, 0.0, 0.0, 1.0]],

[[0.999871, 0.004153, 0.015515, 0.0470],
[-0.004165, 0.999991, 0.000711, 0.0706],
[-0.015511, -0.000776, 0.999879,-1.4049],
[0.0, 0.0, 0.0, 1.0]]])
                extrinsics_vggt_65 = torch.tensor([[[1.000000, -0.000081, 0.000031, 0.000026],
[0.000081, 1.000000, -0.000032, -0.000010],
[-0.000031, 0.000032, 1.000000, -0.000030],
[0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]],

[[0.999993, -0.001011, -0.003500, -0.003342],
[0.001006, 0.999998, -0.001546, 0.002190],
[0.003502, 0.001542, 0.999993, -0.030831],
 [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]])
                extrinsics_umeyama_aligned = torch.tensor([[[ 1.0000e+00, -5.6000e-05, -5.9000e-05,  9.4164e-02],
         [ 5.6000e-05,  1.0000e+00, -2.7000e-05,  6.0083e-02],
         [ 5.9000e-05,  2.7000e-05,  1.0000e+00, -1.2571e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.9987e-01,  4.1530e-03,  1.5515e-02,  5.6808e-02],
         [-4.1650e-03,  9.9999e-01,  7.1100e-04,  6.6878e-02],
         [-1.5511e-02, -7.7600e-04,  9.9988e-01, -1.4141e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]])
                extrinsics_umeyama_aligned = extrinsics_umeyama_aligned.inverse()
                gt_rot_ume=torch.tensor([[[ 9.8662e-01, -1.2380e-02,  1.6258e-01, -9.4164e-02],
         [ 1.1423e-02,  9.9991e-01,  6.8159e-03, -6.0083e-02],
         [-1.6265e-01, -4.8675e-03,  9.8667e-01, 1.2571e+00],
         [-2.7533e-09, -1.2589e-10,  1.6831e-08,  1.0000e+00]],

        [[ 9.8940e-01, -1.6843e-02,  1.4423e-01, -5.6808e-02],
         [ 1.6304e-02,  9.9985e-01,  4.9176e-03, -6.6878e-02],
         [-1.4429e-01, -2.5140e-03,  9.8953e-01,  1.4141e+00],
         [-2.2062e-09,  3.2296e-12, -6.8599e-09,  1.0000e+00]]])
                gt_rot=torch.tensor([[[ 9.8662e-01, -1.2380e-02,  1.6258e-01,  0.000007],
         [ 1.1423e-02,  9.9991e-01,  6.8159e-03, 0.000018],
         [-1.6265e-01, -4.8675e-03,  9.8667e-01, -0.000016],
         [-2.7533e-09, -1.2589e-10,  1.6831e-08,  1.0000e+00]],

        [[ 9.8940e-01, -1.6843e-02,  1.4423e-01, 0.010875],
         [ 1.6304e-02,  9.9985e-01,  4.9176e-03, -0.003780],
         [-1.4429e-01, -2.5140e-03,  9.8953e-01,  0.061710],
         [-2.2062e-09,  3.2296e-12, -6.8599e-09,  1.0000e+00]]])
                # extrinsics_vggt=gt_rot_ume
                extrinsics_vggt=extrinsics_vggt_5a
                extrinsics_vggt=extrinsics_vggt.inverse()
                # choise 1: direct scale translation
                scale_factor = 50.
                extrinsics_vggt[:,:3,3]*=scale_factor
                # choise 2: umeyama_aligned
                # extrinsics_vggt = extrinsics_umeyama_aligned
                intrinsics_vggt = torch.tensor([[[0.66409, 0.000000, 0.5],
[0.000000, 1.176456, 0.5],
[0.000000, 0.000000, 1.000000]],
[[0.626594, 0.000000, 0.5],
[0.000000, 1.1058973, 0.5],
[0.000000, 0.000000, 1.000000]]])
                extrinsics_vggsfm_aligned = torch.tensor([[[ 0.985243668074267, -0.045489101247291074, -0.1650019884343139,  0.10122522816325084],
    [ 0.04305390512425685,  0.9989050066465699,  -0.018307073768568546, 0.06973023819216038],
    [ 0.1656540846858756,   0.010932948556085991, 0.9861233162555051, -1.261752829508251],
    [ 0.0, 0.0, 0.0, 1.0]],
    [[ 0.9882409631442587, -0.04058578814064203, -0.14741978349160156,  0.04918173611951505],
    [ 0.038792520529507594, 0.999134389866521,  -0.015020364077555439, 0.07739309314550169],
    [ 0.14790178874738297,  0.009124954085227295, 0.9889599567718942, -1.4072476016685367],
    [ 0.0, 0.0, 0.0, 1.0]]])
                extrinsics_vggt_aligned = torch.tensor([[[ 0.8878230635721694, -0.43185741203833417, -0.15896346579286688,  0.08259634989411138],
     [ 0.4121766730678942,  0.8998685886012389,  -0.14264260733559453,  0.18309631661530598],
     [ 0.2046474968525386,  0.06112036417076369,  0.9769256384779874,  -1.252372775114746],
     [ 0.0, 0.0, 0.0, 1.0]],

    [[ 0.8927452016380407, -0.4268330022460518, -0.14428996204128686,  0.03746922927063111],
     [ 0.4084353269919033,  0.9018625100553798, -0.1407998459573982,   0.18868511489554554],
     [ 0.19022772830812834,  0.0667652690418602,  0.9794671052324815,  -1.3961770830932392],
     [ 0.0, 0.0, 0.0, 1.0]]])
                
    #             extrinsics_vggsfm_norm = torch.tensor([
    # [[ 0.9982360783774592,  -0.05929450296637074, -0.0029822380123343275, -0.0010664131640255624],
    #  [ 0.05928096121951252,  0.9982314589550724,  -0.00444094464256064,    0.0015391991996283663],
    #  [ 0.003240287407285571, 0.004256321228324572, 0.9999856920312007,     0.0023695330712160856],
    #  [ 0.0, 0.0, 0.0, 1.0]],

    # [[ 0.9984021869073282,  -0.05451143095362205,  0.014885465194377405, -0.03079521723519597],
    #  [ 0.05454545464451762,  0.9985095016335451,  -0.001889054036345525,  0.01289511456540876],
    #  [-0.014760303394151355, 0.002697970147695946, 0.9998874208633663,   -0.1413001655517699],
    #  [ 0.0, 0.0, 0.0, 1.0]]])
                
                extrinsics_vggsfm_norm = torch.tensor([
    [[ 0.9798601905135721,   0.19927744836356182, -0.01274776923284851,   -0.0009618806757808248],
     [-0.19886322396920691,  0.9796232848122979,   0.028136062377958855,   0.002454982475347965],
     [ 0.018094894287589855, -0.0250343449539088,  0.9995228143336446,     0.0022854315369160326],
     [ 0.0, 0.0, 0.0, 1.0]],

    [[ 0.9790808087908461,   0.20340361116816325,  0.005266955592503611,  -0.02486978859596383],
     [-0.2034688352007551,   0.9785989819090871,   0.03073216081804645,    0.01445706027437607],
     [ 0.0010967951088069763, -0.031160930189084078, 0.999513778529461,   -0.1421347680184406],
     [ 0.0, 0.0, 0.0, 1.0]]
])
#                 extrinsics_vggt_norm = torch.tensor([
#     [[ 0.895791175959932, -0.444207597848377, -0.01542008709757408, -0.0031738943947491184],
#      [ 0.4423017345826051, 0.8943009272973241, -0.06778662862515684, 0.0015856712370884453],
#      [ 0.04390153365818695, 0.05390233249981658, 0.997580670368837, 0.0017469006093189289],
#      [ 0.0, 0.0, 0.0, 1.0]],

#     [[ 0.898330962623583, -0.4393193255172301, -0.00010871463892108031, -0.029468125784154534],
#      [ 0.4383385088124405, 0.8963418363414738, -0.06656323396651295, 0.009009377174927767],
#      [ 0.02933996052950176, 0.05974816023176561, 0.997782202720136, -0.1406235645228287],
#      [ 0.0, 0.0, 0.0, 1.0]]
# ])
                extrinsics_vggt_norm = torch.tensor([
    [[ 0.8910361012094945, -0.4110385685910187, 0.19261869450297384, 0.0026600257508241866],
     [ 0.4311957809742515, 0.899037351319993, -0.07617111920890368, -0.0009847216079793355],
     [-0.14186213311306486, 0.1509275854911157, 0.9783128329560198, 0.0024001866634818904],
     [ 0.0, 0.0, 0.0, 1.0]],

    [[ 0.8916727697373612, -0.40304077145370276, 0.20610145136535757, -0.06434147098329834],
     [ 0.4261450762560264, 0.9009470501692893, -0.08182167667556917, 0.014678022142410402],
     [-0.15270902295427136, 0.1607872797744276, 0.9751037918965815, -0.13143028861027364],
     [ 0.0, 0.0, 0.0, 1.0]]
])

                
                # R_vggt = extrinsics_vggt_aligned[:, :3, :3]
                # t_vggt = extrinsics_vggt_aligned[:, :3, 3:4]
                # R_gt_c2w = extrinsics[context_indices][:, :3, :3]
                # R_gt = R_gt_c2w.transpose(-1, -2)
                # C_vggt = -torch.bmm(R_vggt.transpose(1, 2), t_vggt)
                # t_new = -torch.bmm(R_gt, C_vggt)
                # new_w2c = torch.cat([R_gt, t_new], dim=-1)
                # bottom_row = torch.tensor([0, 0, 0, 1], dtype=new_w2c.dtype, device=new_w2c.device).reshape(1, 1, 4)
                # bottom_row = bottom_row.repeat(new_w2c.shape[0], 1, 1)
                # extrinsics_vggt_aligned = torch.cat([new_w2c, bottom_row], dim=1)
                camera_norm_matrix = extrinsics_vggt_aligned[0].unsqueeze(0).inverse()
                extrinsics_vggt_aligned = torch.bmm(camera_norm_matrix.repeat(extrinsics_vggt_aligned.shape[0], 1, 1), extrinsics_vggt_aligned)
                extrinsics_vggt_aligned = extrinsics_vggt_aligned.inverse()
                # camera_norm_matrix = extrinsics_vggt_norm[0].unsqueeze(0).inverse()
                # extrinsics_vggt_norm = torch.bmm(camera_norm_matrix.repeat(extrinsics_vggt_norm.shape[0], 1, 1), extrinsics_vggt_norm)
                extrinsics_vggt_norm = extrinsics_vggt_norm.inverse()
                
                # R_vggsfm = extrinsics_vggsfm_aligned[:, :3, :3]
                # t_vggsfm = extrinsics_vggsfm_aligned[:, :3, 3:4]
                # C_vggsfm = -torch.bmm(R_vggsfm.transpose(1, 2), t_vggsfm)
                # t_new = -torch.bmm(R_gt, C_vggsfm)
                # new_w2c = torch.cat([R_gt, t_new], dim=-1)
                # bottom_row = torch.tensor([0, 0, 0, 1], dtype=new_w2c.dtype, device=new_w2c.device).reshape(1, 1, 4)
                # bottom_row = bottom_row.repeat(new_w2c.shape[0], 1, 1)
                # extrinsics_vggsfm_aligned = torch.cat([new_w2c, bottom_row], dim=1)
                # extrinsics_vggsfm_aligned = extrinsics_vggsfm_aligned.inverse()
                camera_norm_matrix = extrinsics_vggsfm_aligned[0].unsqueeze(0).inverse()
                extrinsics_vggsfm_aligned = torch.bmm(camera_norm_matrix.repeat(extrinsics_vggsfm_aligned.shape[0], 1, 1), extrinsics_vggsfm_aligned)
                extrinsics_vggsfm_aligned = extrinsics_vggsfm_aligned.inverse()
                # camera_norm_matrix = extrinsics_vggsfm_norm[0].unsqueeze(0).inverse()
                # extrinsics_vggsfm_norm = torch.bmm(camera_norm_matrix.repeat(extrinsics_vggsfm_norm.shape[0], 1, 1), extrinsics_vggsfm_norm)
                extrinsics_vggsfm_norm = extrinsics_vggsfm_norm.inverse()
                
                if self.stage=="train" or self.stage=="test":
                    input_extrinsics=extrinsics_vggt_finetune[context_indices]
                else:
                    input_extrinsics=extrinsics[context_indices]
                example = {
                    "context": {
                        "extrinsics": extrinsics_vggt_finetune[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "raw_image": context_images,
                        "raw_extrinsics": extrinsics[context_indices],
                        "raw_intrinsics": intrinsics[context_indices],
                        "near": context_nears,
                        "far": context_fars,
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": extrinsics_vggt_finetune[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": target_nears,
                        "far": target_fars,
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
