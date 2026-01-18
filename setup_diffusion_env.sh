#!/bin/bash

# Setup script to install Diffusion Policy dependencies
# This can be run in your existing booster-ssl environment

echo "Installing Diffusion Policy dependencies into booster-ssl environment..."

# Activate the environment
source /mnt/home/fang/anaconda3/etc/profile.d/conda.sh
conda activate booster-ssl

# Install missing dependencies
echo "Installing zarr and other required packages..."

# Install packages one by one to catch errors
pip install zarr==2.12.0
pip install numcodecs==0.10.2
pip install dill==0.3.5.1
pip install einops==0.4.1

# Install hydra-core (configuration management for diffusion policy)
pip install hydra-core==1.2.0

# Install diffusion-related packages
pip install diffusers==0.11.1

# Install optional dependencies (may not be needed for basic training)
pip install av==10.0.0 || echo "Warning: av installation failed, skipping..."
pip install pymunk==6.2.1 || echo "Warning: pymunk installation failed, skipping..."

echo ""
echo "✓ Installation complete!"
echo ""
echo "You can now test the dataset with:"
echo "  conda activate booster-ssl"
echo "  python test_diffusion_dataset.py"
echo ""
echo "Or start training with:"
echo "  conda activate booster-ssl"
echo "  ./train_diffusion.sh"
