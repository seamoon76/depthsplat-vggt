<p align="center">
  <h1 align="center">Benchmarking Feed-Forward 3DGSh</h1>
  <div align="center"></div>
</p>

## Installation

Our code is developed using PyTorch 2.4.0, CUDA 12.4, and Python 3.10. 

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

## Pre-trained models

Our pre-trained models can be downloaded from this [polybox link(TODO)]().

Put the `pretrained` directory under the root path of this project.


## Datasets

We use Re10k dataset. Firstly, please refer to [DepthSplat's DATASETS.md](https://github.com/cvg/depthsplat/blob/main/DATASETS.md) to download the Re10k dataset.

Secondly, we use [VGGT](https://github.com/facebookresearch/vggt/tree/main) to process the Re10k data to estimate camera poses and depth maps. Please clone the [VGGT](https://github.com/facebookresearch/vggt/tree/main) code and replace its [TODO]() file with our provided [TODO]() file, and store the processed data at `datasets/vggt_re10k`.

We provide a minimal example set of processed training and test data at this [polybox link(TODO)](). You can download the `datasets` directory and put it under the root path of this project for a quick validation.


### Evaluation



#### RealEstate10K

<summary><b>To evaluate our fine-tuned model, use:</b></summary> -->

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

### Training

- Before training, you need to download the pre-trained [UniMatch](https://github.com/autonomousvision/unimatch) and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) weights, and set up your [wandb account](config/main.yaml) (in particular, by setting `wandb.entity=YOUR_ACCOUNT`) for logging.

```
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth -P pretrained
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -P pretrained
```

- By default, we train our models using four GH200 GPUs (96GB VRAM each). However, this is not a strict requirement—our model can be trained on different GPUs as well. For example, we have verified that configurations such as four RTX 4090 GPUs (24GB VRAM each) or a single A100 GPU (80GB VRAM) can achieve very similar results, with a PSNR difference of at most 0.1 dB. Just ensure that the total number of training samples, calculated as (number of GPUs &times; `data_loader.train.batch_size` &times; `trainer.max_steps`), remains the same. Check out the scripts [scripts/re10k_depthsplat_train.sh](scripts/re10k_depthsplat_train.sh) and [scripts/dl3dv_depthsplat_train.sh](scripts/dl3dv_depthsplat_train.sh) for details.



## Depth Prediction

We fine-tune our Gaussian Splatting pre-trained depth model using ground-truth depth supervision. The depth models are trained with a randomly selected number of input images (ranging from 2 to 8) and can be used for depth prediction from multi-view posed images. For more details, please refer to [scripts/inference_depth.sh](scripts/inference_depth.sh).


<p align="center">
  <a href="">
    <img src="https://haofeixu.github.io/depthsplat/assets/depth/img_depth_c31a5a509ab9c526.png" alt="Logo" width="100%">
  </a>
</p>


## Citation

```
@inproceedings{xu2024depthsplat,
      title   = {DepthSplat: Connecting Gaussian Splatting and Depth},
      author  = {Xu, Haofei and Peng, Songyou and Wang, Fangjinhua and Blum, Hermann and Barath, Daniel and Geiger, Andreas and Pollefeys, Marc},
      booktitle={CVPR},
      year={2025}
    }
```



## Acknowledgements

This project is developed with several fantastic repos: [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [MVSplat360](https://github.com/donydchen/mvsplat360), [UniMatch](https://github.com/autonomousvision/unimatch), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [DL3DV](https://github.com/DL3DV-10K/Dataset). We thank the original authors for their excellent work.


