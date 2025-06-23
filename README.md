<p align="center">
  <h1 align="center">Benchmarking Feed-Forward 3DGSh</h1>
  <div align="center"></div>
</p>

## Installation

Our code is developed using PyTorch 2.4.0, CUDA 12.4, Python 3.10 and [colmap 3.11.1](https://colmap.github.io). 

We recommend setting up a virtual environment using either [conda](https://docs.anaconda.com/miniconda/) or [venv](https://docs.python.org/3/library/venv.html) before installation:

```bash
# conda
conda create -y -n depthsplat python=3.10
conda activate depthsplat

# or venv
# python -m venv /path/to/venv/depthsplat
# source /path/to/venv/depthsplat/bin/activate

# installation
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
Install colmap v3.11.1 following [colmap doc](https://colmap.github.io/install.html).

## Pre-trained models

Our pre-trained models can be downloaded from this [polybox link](https://polybox.ethz.ch/index.php/s/2cCrcS2tsAf9RnW).

Put the `pretrained` directory under the root path of this project.


## Datasets

We use Re10k dataset. Firstly, please refer to [DepthSplat's DATASETS.md](https://github.com/cvg/depthsplat/blob/main/DATASETS.md) to download the Re10k dataset.

Secondly, we use calibrated method to get the aligned extrinsics from VGGSfm / VGGT, please store the processed data at `datasets/re10k_vggsfm` or `datasets/re10k_norm`.

We provide a minimal example set of processed training and test data at this [polybox link](https://polybox.ethz.ch/index.php/s/2cCrcS2tsAf9RnW). You can download the `datasets` directory and put it under the root path of this project for a quick validation.

### COLMAP Camera Alignment and Normalization Tools

This repository provides a set of scripts to **align COLMAP camera poses (extrinsics) with ground-truth poses**, particularly for outputs from [VGGSfM](https://github.com/facebookresearch/vggsfm) or [VGGT](https://github.com/facebookresearch/vggt).

#### Usage

##### Method 1: Direct Alignment (Raw COLMAP)

1. Run VGGSfM/VGGT to get raw COLMAP output.
2. Extract extrinsic matrices from COLMAP.
3. Align COLMAP poses with ground truth poses from `.torch` using `alignment.sh`.
4. Write aligned extrinsics to a new `.torch` file using `write_torch.py`.

```bash
# Extract camera intrinsics and extrinsics from COLMAP
bash run_colmap2mvsnet.sh

# Align raw COLMAP extrinsics with GT poses
bash alignment.sh

# Write aligned camera data into .torch file
python write_torch.py --input_folder <output_dir> --output_path <scene>.torch
```

##### Method 2: Normalized Alignment

1. Run VGGSfM/VGGT to get raw  COLMAP and normalize the model with `normalize_model.py`.
2. Extract normalized extrinsics.
3. Align with **normalized** ground-truth poses.
4. Write normalized and aligned extrinsics into `.torch`.

```bash
# Normalize the COLMAP model
python normalize_model.py --input_model <raw_model> --output_model <norm_model>

# Extract camera intrinsics and extrinsics from COLMAP
bash run_colmap2mvsnet.sh

# Align normalized COLMAP with normalized GT
bash alignment.sh

# Write aligned normalized data into .torch file
python write_torch.py --input_folder <norm_output_dir> --output_path <scene>_norm.torch
```



### Evaluation

#### RealEstate10K

We use the same camera views with DepthSplat and NoPoSplat for a fire comparison. 

<details>
<summary>To evaluate our model on the specific camera views, use:</summary> 

```
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/epoch_212-step_277951.ckpt \
mode=test dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
test.save_input_images=true \
test.save_gt_image=true \
test.save_image=true

```
</details>

According to NoPoSplat, we can divide the camera views as high/medium/small/ignore overlap degrees, we use the same view division protocals with NoPoSplat.


<details>
<summary>To evaluate our model on the high camera views, use:</summary> 

```
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/epoch_212-step_277951.ckpt \
mode=test dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_high.json \
test.save_input_images=true \
test.save_gt_image=true \
test.save_image=true
```
</details>


<details>
<summary>To evaluate our model on the medium camera views, use:</summary>

```
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/epoch_212-step_277951.ckpt \
mode=test dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_medium.json \
test.save_input_images=true \
test.save_gt_image=true \
test.save_image=true

```
</details>


<details>
<summary>To evaluate our model on the small camera views, use:</summary>

```
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/epoch_212-step_277951.ckpt \
mode=test dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_small.json \
test.save_input_images=true \
test.save_gt_image=true \
test.save_image=true

```
</details>


<details>
<summary>To evaluate our model on the ignore camera views, use:</summary> 

```
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/epoch_212-step_277951.ckpt \
mode=test dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_ignore.json \
test.save_input_images=true \
test.save_gt_image=true \
test.save_image=true
```
</details>

### Training

- Before training, you need to download the pre-trained [UniMatch](https://github.com/autonomousvision/unimatch) and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) weights and [DepthSplat](https://github.com/cvg/depthsplat), and set up your [wandb account](config/main.yaml) (in particular, by setting `wandb.entity=YOUR_ACCOUNT`) for logging.

```
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth -P pretrained
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -P pretrained
wget https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth -P pretrained
```

If you have access to student-cluster, you can use:
```
batch run.sh
```
Or use:
```
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
data_loader.train.batch_size=1 dataset.test_chunk_interval=10 trainer.max_steps=4800000 \
model.encoder.gaussian_adapter.gaussian_scale_max=0.3 model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
output_dir=checkpoints/re10k-256x256-depthsplat-small checkpointing.every_n_train_steps=100000 \
checkpointing.resume=False
```

## Other branches

We put the calibration-based method into `colmap` branch, and the correspondence loss feature into `corr` branch. Please check these branches' readme file and our report for more information.

## Citation
This code repository is modified from DepthSplat. So we keep the citation information of DepthSplat here.

```
@inproceedings{xu2024depthsplat,
      title   = {DepthSplat: Connecting Gaussian Splatting and Depth},
      author  = {Xu, Haofei and Peng, Songyou and Wang, Fangjinhua and Blum, Hermann and Barath, Daniel and Geiger, Andreas and Pollefeys, Marc},
      booktitle={CVPR},
      year={2025}
    }
```



## Acknowledgements

This project is developed with several fantastic repos: [DepthSplat](https://github.com/cvg/depthsplat/tree/main), [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [MVSplat360](https://github.com/donydchen/mvsplat360), [UniMatch](https://github.com/autonomousvision/unimatch), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [DL3DV](https://github.com/DL3DV-10K/Dataset). We thank the original authors for their excellent work.

