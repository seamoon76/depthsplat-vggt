import torch
import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

view_ids = [58, 70, 133]

def parse_cam_txt(path):
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')
    extr_start = lines.index("extrinsic") + 1
    intr_start = lines.index("intrinsic") + 1
    extrinsic = np.array([[float(x) for x in line.split()] for line in lines[extr_start:extr_start+4]], dtype=np.float32)
    intrinsic = np.array([[float(x) for x in line.split()] for line in lines[intr_start:intr_start+3]], dtype=np.float32)
    return extrinsic[:3], intrinsic

def load_selected_views(cam_dir: Path, scene_key: str):
    extrinsics = []
    intrinsics = []
    for i in view_ids:
        cam_path = cam_dir / f"{scene_key}_{i:03d}_cam.txt"
        if not cam_path.exists():
            raise FileNotFoundError(f"Missing camera file: {cam_path}")
        ext, intr = parse_cam_txt(cam_path)
        extrinsics.append(ext)
        intrinsics.append(intr)
    extrinsics = torch.tensor(np.stack(extrinsics))  # [3, 3, 4]
    intrinsics = torch.tensor(np.stack(intrinsics))  # [3, 3, 3]

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    zeros = torch.zeros(len(view_ids), 2)
    flat_ext = extrinsics.reshape(len(view_ids), -1)
    pose = torch.cat([fx[:, None], fy[:, None], cx[:, None], cy[:, None], zeros, flat_ext], dim=1)  # [3, 18]
    return pose

def general_repack_chunks(index_path, old_root, new_root, cam_root, selected_scenes):
    # Load original index
    with open(index_path, 'r') as f:
        index = json.load(f)

    # Build mapping: chunk_name -> list of scene_keys
    chunk_to_scenes = defaultdict(list)
    for scene_key in selected_scenes:
        if scene_key not in index:
            print(f"[!] {scene_key} not found in index.json")
            continue
        chunk_to_scenes[index[scene_key]].append(scene_key)

    new_root = Path(new_root)
    new_root.mkdir(parents=True, exist_ok=True)

    for chunk_name, scene_keys in chunk_to_scenes.items():
        print(f"\n[Chunk: {chunk_name}]")

        old_chunk_path = Path(old_root) / chunk_name
        new_chunk_path = Path(new_root) / chunk_name

        if not old_chunk_path.exists():
            print(f"[!] Missing original chunk {old_chunk_path}")
            continue

        chunk_data = torch.load(old_chunk_path)
        new_chunk = []

        for scene_key in scene_keys:
            scene_data = [s for s in chunk_data if s["key"] == scene_key]
            if not scene_data:
                print(f"[!] Scene {scene_key} not found in {chunk_name}")
                continue
            scene = scene_data[0]

            variant_dirs = {
                "vggt_camera_aligned": "with_gt_cams",
                "vggt_camera_norm": "with_norm_cams",
                "vggt_camera_norm2": "norm_with_norm_cams"
            }
            for field_name, folder in variant_dirs.items():
                cam_dir = Path(cam_root) / scene_key / folder
                if not cam_dir.exists():
                    print(f"    [!] Missing folder: {cam_dir}")
                    continue
                try:
                    pose_tensor = load_selected_views(cam_dir, scene_key)
                    scene[field_name] = pose_tensor
                    print(f"    [+] Added {field_name}")
                except Exception as e:
                    print(f"    [!] Failed {field_name}: {e}")
                    continue

            new_chunk.append(scene)

        if new_chunk:
            torch.save(new_chunk, new_chunk_path)
            print(f"[✓] Saved {new_chunk_path} with {len(new_chunk)} scenes")
        else:
            print(f"[!] No valid scenes found for {chunk_name}, skipping write.")

    # Copy index file
    new_index_path = new_root / "index.json"
    with open(new_index_path, 'w') as f:
        json.dump({k: v for k, v in index.items() if k in selected_scenes}, f, indent=2)
    print(f"\n[✓] Wrote new index.json with {len(selected_scenes)} scenes to {new_index_path}")

selected_scenes = [
    "1214f2a11a9fc1ed", "6558c5f10d45a929", "a9b3ff60b213e099", "fea544b472e9abd1",
    "656381bea665bf3d", "21e794f71e31becb", "6771a51bf0cfce7f",
    "bc95e5c7e357f1b7", "5aca87f95a9412c6", "84ab392d682f296b", "c48f19e2ffa52523"
]

general_repack_chunks(
    index_path="/work/courses/3dv/35/datasets/re10k/test/index.json",
    old_root="/work/courses/3dv/35/datasets/re10k/test",
    new_root="/work/courses/3dv/35/datasets/re10k_vggsfm/test",
    cam_root="/home/jiaysun/re10k_vggsfm",
    selected_scenes=selected_scenes
)
