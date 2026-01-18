#!/bin/bash

# Train Diffusion Policy on Booster Soccer Dataset
# This script trains a diffusion-based policy using the collected imitation learning dataset

# Activate conda environment (adjust if your environment name is different)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate booster-ssl

cd diffusion_policy

python train.py \
  --config-name=train_diffusion_unet_booster_lowdim \
  training.device=cuda:0 \
  training.num_epochs=3000 \
  dataloader.batch_size=256 \
  dataloader.num_workers=4

# Alternative: Use Diffusion Transformer instead of UNet
# python train.py \
#   --config-name=train_diffusion_transformer_booster_lowdim \
#   training.device=cuda:0 \
#   training.num_epochs=3000 \
#   dataloader.batch_size=256
