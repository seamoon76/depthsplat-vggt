#!/bin/bash
#SBATCH --time=36:00:10
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --output=depthsplat_log.out
source /work/courses/3dv/35/envs/depthsplat/bin/activate
module load cuda/12.6
cd /work/courses/3dv/35/depthsplat/
export WANDB_API_KEY=2870043c3f7b4982303d8e488c953e41efb710c5
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k data_loader.train.batch_size=1 dataset.test_chunk_interval=10 trainer.max_steps=150000 model.encoder.gaussian_adapter.gaussian_scale_max=0.3 model.encoder.upsample_factor=4 model.encoder.lowest_feature_resolution=4 checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth output_dir=checkpoints/re10k-256x256-depthsplat-small
