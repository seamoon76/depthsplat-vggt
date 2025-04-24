#!/bin/bash
#SBATCH --time=03:10
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --output=depthsplat_log.out
source /work/courses/3dv/35/envs/depthsplat/bin/activate
module load cuda/12.4
cd /work/courses/3dv/35/depthsplat/
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x256-view2-fbe87117.pth \
mode=test \
dataset/view_sampler=evaluation
