#!/usr/bin/env bash
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00 --mem-per-cpu=25000M
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=depthsplat_roma_log_3090.out
source ~/.bashrc && conda init && conda activate /cluster/home/zhangdi/miniconda3/envs/foundation_flow && cd /cluster/home/zhangdi/depthsplat-vggt && \
export CUDA_HOME=/cluster/home/zhangdi/miniconda3/envs/foundation_flow && \
ml stack/2024-06  gcc/12.2.0 && \
ml cuda/12.4.1 && \
export WANDB_API_KEY=79725814960c863c14671b2cf92163c461975aa8 && \

python -m src.main +experiment=re10k data_loader.train.batch_size=1 dataset.test_chunk_interval=10 trainer.max_steps=4800000 model.encoder.gaussian_adapter.gaussian_scale_max=0.3 model.encoder.upsample_factor=4 model.encoder.lowest_feature_resolution=4 checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth output_dir=/cluster/scratch/zhangdi/depthsplat-vggt/re10k-256x256-depthsplat-romaindoor-newcamerahead checkpointing.every_n_train_steps=100000 checkpointing.resume=False 
# python -m src.main +experiment=re10k \ 
# data_loader.train.batch_size=4 \
# dataset.test_chunk_interval=10 \
# trainer.max_steps=4800000 \
# model.encoder.gaussian_adapter.gaussian_scale_max=0.3 \ 
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=4 \
# checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
# output_dir=checkpoints/re10k-256x256-depthsplat-small \
# checkpointing.every_n_train_steps=100000 \
# checkpointing.resume=False \
