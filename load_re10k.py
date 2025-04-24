import torch
import json
import os
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import torchvision.transforms as tf

def convert_image(image): # Float[Tensor, "batch 3 height width"]
    image = Image.open(BytesIO(image.numpy().tobytes()))
    return tf.ToTensor()(image)

def load_scene_images(scene_name, dataset_root, output_dir):
    dataset_root = Path(dataset_root)
    index_file = dataset_root / "index.json"
    
    # Load scene-to-file mapping
    with open(index_file, "r") as f:
        index_mapping = json.load(f)
    
    if scene_name not in index_mapping:
        raise ValueError(f"Scene {scene_name} not found in index file.")
    
    torch_file = dataset_root / index_mapping[scene_name]
    
    if not torch_file.exists():
        raise FileNotFoundError(f"Torch file {torch_file} not found.")
    
    # Load .torch file
    chunk = torch.load(torch_file)
    
    # Find relevant scene data
    scene_data = [item for item in chunk if item["key"] == scene_name]
    if not scene_data:
        raise ValueError(f"Scene {scene_name} not found in {torch_file}.")
    
    scene_data = scene_data[0]  # Assuming one match
    images = scene_data["images"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.ToPILImage()

    # Save given index images
    for idx in view_idx:
        img = transform(convert_image(images[idx]))
        img.save(os.path.join(output_dir, f"{scene_name}_{idx:03d}.png"))
    
    print(f"Saved {len(view_idx)} images for scene {scene_name} in {output_dir}")

# Example usage
scene_name = "28e8300e004ab30b"
dataset_root = "/work/courses/3dv/35/re10k/test"  # Adjust this to your dataset root
output_dir = f"re10k_images/{scene_name}"
view_idx = [58,70]

load_scene_images(scene_name, dataset_root, output_dir)

