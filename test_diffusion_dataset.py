"""
Quick test script to verify the Booster dataset loads correctly for Diffusion Policy.
Run this before starting training to catch any issues early.
"""

import sys
sys.path.append('diffusion_policy')

import numpy as np
from diffusion_policy.dataset.booster_lowdim_dataset import BoosterLowdimDataset

def test_dataset():
    print("=" * 60)
    print("Testing Booster Lowdim Dataset for Diffusion Policy")
    print("=" * 60)
    
    # Test parameters
    dataset_path = "booster_dataset/imitation_learning/booster_soccer_showdown.npz"
    horizon = 16
    n_obs_steps = 2
    n_latency_steps = 0
    n_action_steps = 8
    
    print(f"\nDataset Configuration:")
    print(f"  - Path: {dataset_path}")
    print(f"  - Horizon: {horizon}")
    print(f"  - Obs steps: {n_obs_steps}")
    print(f"  - Action steps: {n_action_steps}")
    
    # Create dataset
    print("\n" + "-" * 60)
    print("Loading dataset...")
    print("-" * 60)
    
    dataset = BoosterLowdimDataset(
        dataset_path=dataset_path,
        horizon=horizon,
        pad_before=n_obs_steps - 1 + n_latency_steps,
        pad_after=n_action_steps - 1,
        seed=42,
        val_ratio=0.05
    )
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  - Total sequences: {len(dataset)}")
    print(f"  - Number of episodes: {dataset.replay_buffer.n_episodes}")
    
    # Test train/val split
    val_dataset = dataset.get_validation_dataset()
    print(f"  - Train sequences: {len(dataset)}")
    print(f"  - Val sequences: {len(val_dataset)}")
    
    # Test data loading
    print("\n" + "-" * 60)
    print("Testing data loading...")
    print("-" * 60)
    
    sample = dataset[0]
    print(f"\n✓ Sample loaded successfully!")
    print(f"  Keys: {list(sample.keys())}")
    print(f"  Obs shape: {sample['obs'].shape}")
    print(f"  Action shape: {sample['action'].shape}")
    print(f"  Obs dtype: {sample['obs'].dtype}")
    print(f"  Action dtype: {sample['action'].dtype}")
    
    # Verify shapes
    expected_obs_shape = (horizon, 89)
    expected_action_shape = (horizon, 12)
    
    assert sample['obs'].shape == expected_obs_shape, \
        f"Obs shape mismatch! Expected {expected_obs_shape}, got {sample['obs'].shape}"
    assert sample['action'].shape == expected_action_shape, \
        f"Action shape mismatch! Expected {expected_action_shape}, got {sample['action'].shape}"
    
    print("\n✓ Shapes verified!")
    
    # Test normalizer
    print("\n" + "-" * 60)
    print("Testing normalizer...")
    print("-" * 60)
    
    normalizer = dataset.get_normalizer()
    print(f"\n✓ Normalizer created successfully!")
    print(f"  Normalizer type: {type(normalizer).__name__}")
    
    # Test batch loading
    print("\n" + "-" * 60)
    print("Testing batch loading...")
    print("-" * 60)
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    batch = next(iter(dataloader))
    print(f"\n✓ Batch loaded successfully!")
    print(f"  Batch obs shape: {batch['obs'].shape}")
    print(f"  Batch action shape: {batch['action'].shape}")
    
    # Statistics
    print("\n" + "-" * 60)
    print("Data Statistics")
    print("-" * 60)
    
    print(f"\nObservations:")
    print(f"  Min: {sample['obs'].min().item():.4f}")
    print(f"  Max: {sample['obs'].max().item():.4f}")
    print(f"  Mean: {sample['obs'].mean().item():.4f}")
    print(f"  Std: {sample['obs'].std().item():.4f}")
    
    print(f"\nActions:")
    print(f"  Min: {sample['action'].min().item():.4f}")
    print(f"  Max: {sample['action'].max().item():.4f}")
    print(f"  Mean: {sample['action'].mean().item():.4f}")
    print(f"  Std: {sample['action'].std().item():.4f}")
    
    # Episode length statistics
    episode_lengths = []
    for i in range(dataset.replay_buffer.n_episodes):
        episode_lengths.append(len(dataset.replay_buffer.get_episode(i)['obs']))
    
    print(f"\nEpisode Lengths:")
    print(f"  Min: {min(episode_lengths)}")
    print(f"  Max: {max(episode_lengths)}")
    print(f"  Mean: {np.mean(episode_lengths):.2f}")
    print(f"  Median: {np.median(episode_lengths):.2f}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now start training with:")
    print("  ./train_diffusion.sh")
    print("\nOr:")
    print("  cd diffusion_policy")
    print("  python train.py --config-name=train_diffusion_unet_booster_lowdim")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_dataset()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
