import torch
import json
import os
from pathlib import Path
from einops import rearrange, repeat
import numpy as np

view_idx = [58, 70, 133]
frame_ids = ["058", "070", "133"]

def convert_poses(poses):  # poses: [N, 18]
    b, _ = poses.shape
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c, intrinsics  # 返回 C2W

def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor):
    camera_norm_matrix = torch.inverse(pivotal_pose)  # [1, 4, 4]
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)
    return poses

def get_camera_center(T):
    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    C = -R.T @ t
    return C.flatten()

def write_normalized_ref_images(dataset_root, output_root):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    index_file = dataset_root / "index.json"
    with open(index_file, "r") as f:
        index_mapping = json.load(f)

    scene_ids = [d.name for d in output_root.iterdir() if d.is_dir()]
    
    for scene_id in scene_ids:
        print(f"Processing scene: {scene_id}")
        if scene_id not in index_mapping:
            print(f"Scene {scene_id} not in index.json")
            continue

        torch_file = dataset_root / index_mapping[scene_id]
        if not torch_file.exists():
            print(f"Missing file for scene {scene_id}: {torch_file}")
            continue

        chunk = torch.load(torch_file)
        scene_data = [item for item in chunk if item["key"] == scene_id]
        if not scene_data:
            print(f"Scene {scene_id} not found in torch file")
            continue

        scene_data = scene_data[0]
        poses = scene_data["cameras"]

        if max(view_idx) >= len(poses):
            print(f"Scene {scene_id} has only {len(poses)} views, skipping.")
            continue

        extrinsics, _ = convert_poses(poses)  # [N, 4, 4]

        # normalize the extrinsics
        # pivotal_pose = extrinsics[view_idx[0]].unsqueeze(0)  # [1, 4, 4]
        # norm_extrinsics = camera_normalization(pivotal_pose, extrinsics)

        scene_output_dir = output_root / scene_id
        ref_txt_path = scene_output_dir / "ref_images.txt"

        with open(ref_txt_path, "w") as f:
            for i, frame_id in enumerate(frame_ids):
                cam_idx = view_idx[i]
                # T = norm_extrinsics[cam_idx]
                T = extrinsics[cam_idx]
                center = get_camera_center(T)
                image_name = f"{scene_id}_{frame_id}.png"
                f.write(f"{image_name} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n")

                print(f"\nExtrinsics for {scene_id} view {frame_id}:\n{T.numpy()}")

        print(f"Saved centers to {ref_txt_path}")

# 调用
dataset_root = "/work/courses/3dv/35/datasets/re10k/test"
output_root = "/home/jiaysun/re10k_vggsfm"
write_normalized_ref_images(dataset_root, output_root)
