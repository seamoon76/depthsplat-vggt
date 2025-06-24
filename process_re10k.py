import torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import os
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from torchvision import transforms as TF


def invert_convert_poses(extrinsics, intrinsics, resolution): # output 18 dim tensor
    H, W = resolution
    b = extrinsics.shape[0]

    # Extract normalized intrinsics
    fx = intrinsics[:, 0, 0] / W
    fy = intrinsics[:, 1, 1] / H
    cx = intrinsics[:, 0, 2] / W
    cy = intrinsics[:, 1, 2] / H
    s1 = intrinsics[:, 0, 1] / W  # skew x
    s2 = intrinsics[:, 1, 0] / H  # skew y (usually 0, but you might encode something here)

    # Build pose vector
    pose_vec = torch.zeros((b, 18), dtype=torch.float32, device=extrinsics.device)
    pose_vec[:, 0] = fx
    pose_vec[:, 1] = fy
    pose_vec[:, 2] = cx
    pose_vec[:, 3] = cy
    pose_vec[:, 4] = s1
    pose_vec[:, 5] = s2
    pose_vec[:, 6:] = extrinsics.reshape(b, -1)

    return pose_vec

def convert_image(image): # Float[Tensor, "batch 3 height width"]
    image = Image.open(BytesIO(image.numpy().tobytes()))
    return image

def preprocess_images(image_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_list (list): List of Image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_list) == 0:
        raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for img in image_list:

        # Open image
        # img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
        
        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

# Example usage
dataset_root = "/cluster/project/cvg/haofei/datasets/depthsplat/re10k/train"  # Adjust this to your dataset root
save_root = "datasets/vggt_re10k"
os.makedirs(save_root, exist_ok=True)

# Load the vggt model
device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

dataset_root = Path(dataset_root)
transform = transforms.ToPILImage()

torch_files = [f for f in os.listdir(dataset_root) if f.endswith(".torch")]
process_torch = sorted(torch_files, key=lambda x: int(x.split(".")[0]))

for file in process_torch:
    torch_file = os.path.join(dataset_root, file)
    print('processing', torch_file)
    chunk = torch.load(torch_file)

    for scene_data in chunk:
        scene_name = scene_data['key']
        images_list = scene_data["images"]
        print(f'{len(images_list)} images in {scene_name}')
        all_imgs = [convert_image(img) for img in images_list]

        images = preprocess_images(all_imgs[:2]).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # Add batch dimension
                B, V, C, H, W = images.shape
                aggregated_tokens_list, ps_idx = model.aggregator(images)
                
            # Predict cameras (extrinsic and intrinsic matrices)
            breakpoint()
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            # Predict Depth Maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

        # convert camera params
        vggt_camera = invert_convert_poses(extrinsic[0], intrinsic[0], (H,W)) # [143, 18]

        scene_data['vggt_camera'] = vggt_camera

        depth_max = depth_map[0,:,:,:,0].amax(dim=(1,2))
        depth_min = depth_map[0,:,:,:,0].amin(dim=(1,2))

        scene_data['depth'] = torch.stack([depth_max, depth_min], dim=1)

        del aggregated_tokens_list, images, depth_map, depth_conf  # free memory
        torch.cuda.empty_cache()

    new_torch = os.path.join(save_root, file)
    torch.save(chunk, new_torch)

    # check if the new param saved
    print('save new torch file at', new_torch)
    print('all keys in scene', chunk[0].keys())