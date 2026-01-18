#!/bin/bash

# Minimal dependency installation for Diffusion Policy
# This installs only the absolutely required packages

echo "Installing minimal dependencies for Diffusion Policy..."

# Activate the environment
source /mnt/home/fang/anaconda3/etc/profile.d/conda.sh
conda activate booster-ssl

# Check if packages are already installed
echo ""
echo "Checking existing packages..."
python -c "import zarr; print('✓ zarr already installed')" 2>/dev/null || pip install zarr
python -c "import hydra; print('✓ hydra-core already installed')" 2>/dev/null || pip install hydra-core
python -c "import diffusers; print('✓ diffusers already installed')" 2>/dev/null || pip install diffusers
python -c "import einops; print('✓ einops already installed')" 2>/dev/null || pip install einops
python -c "import dill; print('✓ dill already installed')" 2>/dev/null || pip install dill

echo ""
echo "✓ Core dependencies installed!"
echo ""
echo "Testing dataset loader..."
python -c "
import sys
sys.path.append('diffusion_policy')
try:
    from diffusion_policy.dataset.booster_lowdim_dataset import BoosterLowdimDataset
    print('✓ Dataset loader imports successfully!')
except ImportError as e:
    print(f'✗ Import error: {e}')
    print('You may need to install additional dependencies.')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete! You can now:"
    echo "   python test_diffusion_dataset.py"
    echo "   ./train_diffusion.sh"
else
    echo ""
    echo "⚠️  There were some issues. Try running the full setup:"
    echo "   ./setup_diffusion_env.sh"
fi
