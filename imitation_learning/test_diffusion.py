"""
Test script for Diffusion Policy model on Booster Soccer environments.
Usage:
    python imitation_learning/test_diffusion.py \
        --checkpoint diffusion_policy/data/outputs/2026.01.15/20.25.01_train_diffusion_unet_booster_lowdim_booster_lowdim/checkpoints/latest.ckpt \
        --env_name LowerT1GoaliePenaltyKick-v0 \
        --device cuda:0
"""

import os
import sys
import argparse
import pathlib
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf

# Save original working directory
original_cwd = os.getcwd()

# Add diffusion_policy to path (must be before importing diffusion_policy)
# Follow the same pattern as diffusion_policy workspace files
DIFFUSION_POLICY_DIR = str(pathlib.Path(__file__).parent.parent / "diffusion_policy")
if DIFFUSION_POLICY_DIR not in sys.path:
    sys.path.insert(0, DIFFUSION_POLICY_DIR)

# Change to diffusion_policy directory for hydra config resolution
# (similar to how train.py and workspace files work)
if os.path.exists(DIFFUSION_POLICY_DIR):
    os.chdir(DIFFUSION_POLICY_DIR)

# Make repo root importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import gymnasium as gym
import sai_mujoco  # noqa: F401  # registers envs
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from booster_control.t1_utils import LowerT1JoyStick
from imitation_learning.scripts.preprocessor import Preprocessor


def get_task_one_hot(env_name):
    """Get task one-hot encoding based on environment name."""
    if "GoaliePenaltyKick" in env_name:
        return np.array([1.0, 0.0, 0.0])
    elif "ObstaclePenaltyKick" in env_name:
        return np.array([0.0, 1.0, 0.0])
    elif "KickToTarget" in env_name:
        return np.array([0.0, 0.0, 1.0])
    else:
        return np.array([1.0, 0.0, 0.0])  # default


def main(args):
    # Ensure checkpoint path is absolute
    if not os.path.isabs(args.checkpoint):
        # Try relative to original working directory first
        orig_path = os.path.join(original_cwd, args.checkpoint)
        # Then try relative to diffusion_policy directory (current working dir)
        curr_path = os.path.join(os.getcwd(), args.checkpoint)
        
        if os.path.exists(orig_path):
            checkpoint_path = orig_path
        elif os.path.exists(curr_path):
            checkpoint_path = curr_path
        else:
            checkpoint_path = args.checkpoint
            print(f"Warning: Checkpoint path not found, using as-is: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # Create workspace
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    # Move to device and set to eval mode
    device = torch.device(args.device)
    policy.to(device)
    policy.eval()
    print(f"Policy loaded on device: {device}")
    
    # Create environment
    print(f"Creating environment: {args.env_name}")
    env_kwargs = {}
    if args.render:
        env_kwargs["render_mode"] = "human"
        if args.renderer:
            env_kwargs["renderer"] = args.renderer
    else:
        env_kwargs["render_mode"] = None
    env = gym.make(args.env_name, **env_kwargs)
    lower_t1_robot = LowerT1JoyStick(env.unwrapped)
    lower_t1_robot.reset(env.unwrapped)  # Initialize the controller
    preprocessor = Preprocessor()
    task_one_hot = get_task_one_hot(args.env_name)
    
    # Get policy parameters
    n_obs_steps = cfg.policy.n_obs_steps
    n_action_steps = cfg.policy.n_action_steps
    obs_dim = cfg.task.obs_dim
    
    print(f"Policy config: n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}, obs_dim={obs_dim}")
    
    # Load a sample observation from dataset for comparison
    dataset_obs_sample = None
    try:
        dataset_path = os.path.join(repo_root, "booster_dataset/imitation_learning/booster_soccer_showdown.npz")
        if os.path.exists(dataset_path):
            dataset_sample = np.load(dataset_path, allow_pickle=True)
            dataset_obs_sample = dataset_sample["observations"][0]
            print(f"\nDataset observation sample shape: {dataset_obs_sample.shape}")
            print(f"Dataset observation range: [{dataset_obs_sample.min():.3f}, {dataset_obs_sample.max():.3f}]")
            print(f"Dataset observation first 12 (qpos): {dataset_obs_sample[:12]}")
        else:
            print(f"Dataset not found at {dataset_path}, skipping comparison")
    except Exception as e:
        print(f"Could not load dataset sample for comparison: {e}")
    
    # Initialize observation buffer and action sequence
    obs_buffer = []
    action_sequence = None
    action_step_idx = 0
    
    episode_count = 0
    total_reward = 0.0
    
    # Main loop
    observation, info = env.reset()
    step_count = 0
    return_list = []
    success_list = []
    step_count_list = []
    while episode_count < args.episodes:
        # Preprocess observation
        preprocessed_obs = preprocessor.modify_state(observation.copy(), info.copy(), task_one_hot)
        
        # Add to observation buffer
        obs_buffer.append(preprocessed_obs.copy())
        if len(obs_buffer) > n_obs_steps:
            obs_buffer.pop(0)
        
        # Pad if needed (for first few steps)
        if len(obs_buffer) < n_obs_steps:
            # Pad with the first observation
            while len(obs_buffer) < n_obs_steps:
                obs_buffer.insert(0, obs_buffer[0] if len(obs_buffer) > 0 else preprocessed_obs.copy())
        
        # Predict action (only when we need a new prediction or don't have one)
        # The policy predicts n_action_steps actions, we should use them sequentially
        if action_sequence is None or action_step_idx >= n_action_steps:
            # Prepare observation for policy
            obs_dict = {
                'obs': torch.from_numpy(np.stack(obs_buffer)).float().unsqueeze(0).to(device)  # [1, n_obs_steps, obs_dim]
            }
            
            # Time to get a new action prediction
            if args.test_random:
                # Test with random actions to verify control pipeline works
                action_sequence = np.random.uniform(-1.0, 1.0, size=(n_action_steps, 12))
                if step_count == 0:
                    print("WARNING: Using random actions for testing!")
            else:
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                    action_sequence = action_dict['action'].cpu().numpy()[0]  # [n_action_steps, action_dim]
                    # print('action_sequence: ', action_sequence.shape)
            
            # Reset action step index
            action_step_idx = 0
            
            # Debug: Check if actions are all zeros or very small (only on first prediction)
            if step_count == 0 or (action_sequence is not None and step_count % (n_action_steps * 5) == 0):
                print(f"\n=== Policy Prediction (step {step_count}) ===")
                # print(f"Action sequence shape: {action_sequence.shape}")
                # print(f"Action stats: min={action_sequence.min():.6f}, max={action_sequence.max():.6f}, mean={action_sequence.mean():.6f}, std={action_sequence.std():.6f}")
                # print(f"Are actions all zeros? {np.allclose(action_sequence, 0)}")
                if step_count == 0:
                    print(f"Observation stats: min={preprocessed_obs.min():.3f}, max={preprocessed_obs.max():.3f}, mean={preprocessed_obs.mean():.3f}")
                    # print(f"Observation first 12 (qpos): {preprocessed_obs[:12]}")
                    # print(f"Observation next 12 (qvel): {preprocessed_obs[12:24]}")
                    
                    # Compare with dataset sample if available
                    # if dataset_obs_sample is not None:
                        # print(f"\nComparison with dataset sample:")
                        # print(f"  Dataset qpos: {dataset_obs_sample[:12]}")
                        # print(f"  Test qpos:    {preprocessed_obs[:12]}")
                        # print(f"  Match: {np.allclose(dataset_obs_sample[:12], preprocessed_obs[:12], atol=0.1)}")
                    
                    # Check normalizer stats if available
                    if hasattr(policy, 'normalizer'):
                        try:
                            # Check if normalizer has 'obs' key safely
                            if hasattr(policy.normalizer, 'params_dict') and 'obs' in policy.normalizer.params_dict:
                                print(f"Policy has observation normalizer")
                        except Exception as e:
                            pass
        
        # Select which action step to use from the sequence
        action_step = action_sequence[action_step_idx]  # [action_dim]
        action_step_idx += 1
        
        # Convert action to torque using LowerT1JoyStick
        action_dim = len(action_step)
        
        if action_dim == 3:
            # High-level command (vx, vy, yaw_rate) - convert to torques
            ctrl, _ = lower_t1_robot.get_actions(action_step, observation, info)
        elif action_dim == 12:
            # Normalized actions from low-level policy (range [-1, 1])
            # Apply action scaling if specified
            if args.action_scale != 1.0:
                action_step = action_step * args.action_scale
                if step_count == 0:
                    print(f"Applying action scale: {args.action_scale}")
            
            # Need to clip to ensure they're in the right range after scaling
            action_step = np.clip(action_step, -1.0, 1.0)
            
            # Convert normalized actions to torques
            ctrl = lower_t1_robot.get_torque(observation, action_step)
        else:
            raise ValueError(f"Unexpected action dimension: {action_dim}. Expected 3 (commands) or 12 (normalized actions)")
        
        # Detailed debug output for first few steps
        if step_count < 10 or step_count % 50 == 0:
            print(f"\nStep {step_count} (action_step_idx={action_step_idx-1}/{n_action_steps-1}):")
            # print(f"  Action shape: {action_step.shape}, range: [{action_step.min():.3f}, {action_step.max():.3f}], mean: {action_step.mean():.3f}")
            # print(f"  Ctrl shape: {ctrl.shape}, range: [{ctrl.min():.3f}, {ctrl.max():.3f}], mean: {ctrl.mean():.3f}")
            # print(f"  Observation qpos (first 12): {observation[:12]}")
            # print(f"  Observation qvel (next 12): {observation[12:24]}")
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(ctrl)
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            episode_count += 1
            print(f"Episode {episode_count}/{args.episodes} completed. "
                  f"Steps: {step_count}, Total Reward: {total_reward:.2f}, "
                  f"Success: {info.get('success', False)}")
            
            return_list.append(total_reward)
            success_list.append(info.get('success', False))
            step_count_list.append(step_count)
            # Don't break if we've completed all episodes
            if episode_count >= args.episodes:
                break
            
            # Reset for new episode
            total_reward = 0.0
            step_count = 0
            
            # Reset controller BEFORE environment reset to ensure clean state
            lower_t1_robot.reset(env.unwrapped)  # Reset the controller for new episode
            
            # Reset environment
            observation, info = env.reset()
            
            # Reset buffers and action sequence
            obs_buffer = []  # Reset observation buffer
            action_sequence = None  # Reset action sequence - will trigger new prediction
            action_step_idx = 0  # Reset action step index
            
            print(f"  Resetting episode {episode_count+1}...")
            
            # Continue to next iteration to process the reset observation
            continue
    
    print(f"Return list: {return_list}, mean: {np.mean(return_list)}, std: {np.std(return_list)}")
    print(f"Step count list: {step_count_list}, mean: {np.mean(step_count_list)}, std: {np.std(step_count_list)}")
    print(f"Success list: {success_list}, mean: {np.mean(success_list)}, std: {np.std(success_list)}")
    
    
    env.close()
    print("Testing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Diffusion Policy on Booster Soccer environments")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to diffusion policy checkpoint (.ckpt file)')
    parser.add_argument('--env_name', type=str, default='LowerT1GoaliePenaltyKick-v0',
                        help='Environment name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on (cuda:0, cpu, etc.)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', default=True,
                        help='Enable rendering (default: True)')
    parser.add_argument('--no-render', dest='render', action='store_false',
                        help='Disable rendering')
    parser.add_argument('--renderer', type=str, default='mujoco',
                        help='Renderer to use (mujoco, mjviewer, etc.)')
    parser.add_argument('--test_random', action='store_true',
                        help='Test with random actions instead of policy (for debugging)')
    parser.add_argument('--action_scale', type=float, default=1.0,
                        help='Scale factor for actions (default: 1.0, try 1.5-2.0 if actions are too small)')
    
    args = parser.parse_args()
    
    main(args)

