import torch.nn.functional as F
import numpy as np
import sys
import os

from sai_rl import SAIClient

from ddpg import DDPG_FF
from training import training_loop
from ppo import PPO

import argparse

# Use gymnasium's built-in vector environments
try:
    from gymnasium.vector import SyncVectorEnv
except ImportError:
    # Fallback to old gym if gymnasium.vector is not available
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'diffusion_policy', 'diffusion_policy', 'gym_util'))
    from sync_vector_env import SyncVectorEnv

## Initialize the SAI client
# Option 1: Use environment variable (recommended)
#   Set it in your shell: export SAI_API_KEY="your-api-key-here"
#   Or add to ~/.bashrc: echo 'export SAI_API_KEY="your-api-key-here"' >> ~/.bashrc
# Option 2: Pass directly (less secure, not recommended for production)
#   sai = SAIClient(comp_id="cmp_xnSCxcJXQclQ", api_key="your-api-key-here")

import os
api_key = os.environ.get("SAI_API_KEY")
if api_key:
    sai = SAIClient(comp_id="cmp_xnSCxcJXQclQ", api_key=api_key)
else:
    # Will try to use default authentication if available
    sai = SAIClient(comp_id="cmp_xnSCxcJXQclQ")

## Make the environment(s)
# Set number of parallel environments (default: 1 for single env, 4 for parallel)
num_envs = int(os.environ.get("NUM_ENVS", "1"))
parser = argparse.ArgumentParser(description='Train RL agent (DDPG or PPO)')

# Algorithm selection
parser.add_argument('--alg', type=str, default='ppo', choices=['ppo', 'ddpg'],
                    help='Algorithm to use: ppo or ddpg (default: ppo)')

# Training parameters
parser.add_argument('--timesteps', type=int, default=1000,
                    help='Number of timesteps to train (default: 1000)')

# Environment parameters
parser.add_argument('--num_envs', type=int, default=1,
                    help='Number of parallel environments (default: 1)')

args = parser.parse_args()
num_envs = args.num_envs

# Create a single environment first to get action/observation spaces for model creation
single_env = sai.make_env()
single_action_space = single_env.action_space
single_obs_space = single_env.observation_space
single_env.close()

if num_envs > 1:
    # Create multiple environments and wrap in VectorEnv
    def make_env_fn():
        return sai.make_env()
    
    env_fns = [make_env_fn for _ in range(num_envs)]
    env = SyncVectorEnv(env_fns)
    print(f"Created {num_envs} parallel environments")
else:
    env = sai.make_env()
    print("Using single environment")

class Preprocessor():

    def get_task_onehot(self, info):
        # Handle both single info dict and batched info dict
        if isinstance(info, dict):
            if 'task_index' in info:
                task_idx = info['task_index']
                if len(task_idx.shape) == 1 and task_idx.shape[0] > 1:
                    # Already batched
                    return task_idx
                elif len(task_idx.shape) == 0:
                    # Scalar, expand
                    return np.expand_dims(task_idx, axis=0)
                else:
                    return task_idx
            else:
                return np.array([])
        else:
            # List of info dicts - should be handled by modify_state
            return np.array([])

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)    
        return a - b + c 

    def modify_state(self, obs, info):
        
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        # Handle both single info dict and list of info dicts (for vectorized envs)
        if isinstance(info, list):
            # Convert list of info dicts to batched format
            batched_info = {}
            for key in info[0].keys():
                batched_info[key] = np.array([inf[key] for inf in info])
        else:
            batched_info = info

        task_onehot = self.get_task_onehot(batched_info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)
        
        if len(batched_info["robot_quat"].shape) == 1:
            batched_info["robot_quat"] = np.expand_dims(batched_info["robot_quat"], axis = 0)
            batched_info["robot_gyro"] = np.expand_dims(batched_info["robot_gyro"], axis = 0)
            batched_info["robot_accelerometer"] = np.expand_dims(batched_info["robot_accelerometer"], axis = 0)
            batched_info["robot_velocimeter"] = np.expand_dims(batched_info["robot_velocimeter"], axis = 0)
            batched_info["goal_team_0_rel_robot"] = np.expand_dims(batched_info["goal_team_0_rel_robot"], axis = 0)
            batched_info["goal_team_1_rel_robot"] = np.expand_dims(batched_info["goal_team_1_rel_robot"], axis = 0)
            batched_info["goal_team_0_rel_ball"] = np.expand_dims(batched_info["goal_team_0_rel_ball"], axis = 0)
            batched_info["goal_team_1_rel_ball"] = np.expand_dims(batched_info["goal_team_1_rel_ball"], axis = 0)
            batched_info["ball_xpos_rel_robot"] = np.expand_dims(batched_info["ball_xpos_rel_robot"], axis = 0) 
            batched_info["ball_velp_rel_robot"] = np.expand_dims(batched_info["ball_velp_rel_robot"], axis = 0) 
            batched_info["ball_velr_rel_robot"] = np.expand_dims(batched_info["ball_velr_rel_robot"], axis = 0) 
            batched_info["player_team"] = np.expand_dims(batched_info["player_team"], axis = 0)
            batched_info["goalkeeper_team_0_xpos_rel_robot"] = np.expand_dims(batched_info["goalkeeper_team_0_xpos_rel_robot"], axis = 0)
            batched_info["goalkeeper_team_0_velp_rel_robot"] = np.expand_dims(batched_info["goalkeeper_team_0_velp_rel_robot"], axis = 0)
            batched_info["goalkeeper_team_1_xpos_rel_robot"] = np.expand_dims(batched_info["goalkeeper_team_1_xpos_rel_robot"], axis = 0)
            batched_info["goalkeeper_team_1_velp_rel_robot"] = np.expand_dims(batched_info["goalkeeper_team_1_velp_rel_robot"], axis = 0)
            batched_info["target_xpos_rel_robot"] = np.expand_dims(batched_info["target_xpos_rel_robot"], axis = 0)
            batched_info["target_velp_rel_robot"] = np.expand_dims(batched_info["target_velp_rel_robot"], axis = 0)
            batched_info["defender_xpos"] = np.expand_dims(batched_info["defender_xpos"], axis = 0)
        
        robot_qpos = obs[:,:12]
        robot_qvel = obs[:,12:24]
        quat = batched_info["robot_quat"]
        base_ang_vel = batched_info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        obs = np.hstack((robot_qpos, 
                         robot_qvel,
                         project_gravity,
                         base_ang_vel,
                         batched_info["robot_accelerometer"],
                         batched_info["robot_velocimeter"],
                         batched_info["goal_team_0_rel_robot"], 
                         batched_info["goal_team_1_rel_robot"], 
                         batched_info["goal_team_0_rel_ball"], 
                         batched_info["goal_team_1_rel_ball"], 
                         batched_info["ball_xpos_rel_robot"], 
                         batched_info["ball_velp_rel_robot"], 
                         batched_info["ball_velr_rel_robot"], 
                         batched_info["player_team"], 
                         batched_info["goalkeeper_team_0_xpos_rel_robot"], 
                         batched_info["goalkeeper_team_0_velp_rel_robot"], 
                         batched_info["goalkeeper_team_1_xpos_rel_robot"], 
                         batched_info["goalkeeper_team_1_velp_rel_robot"], 
                         batched_info["target_xpos_rel_robot"], 
                         batched_info["target_velp_rel_robot"], 
                         batched_info["defender_xpos"],
                         task_onehot))
        
        robot_qpos = obs[:,:12]
        robot_qvel = obs[:,12:24]
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        obs = np.hstack((robot_qpos, 
                         robot_qvel,
                         project_gravity,
                         base_ang_vel,
                         info["robot_accelerometer"],
                         info["robot_velocimeter"],
                         info["goal_team_0_rel_robot"], 
                         info["goal_team_1_rel_robot"], 
                         info["goal_team_0_rel_ball"], 
                         info["goal_team_1_rel_ball"], 
                         info["ball_xpos_rel_robot"], 
                         info["ball_velp_rel_robot"], 
                         info["ball_velr_rel_robot"], 
                         info["player_team"], 
                         info["goalkeeper_team_0_xpos_rel_robot"], 
                         info["goalkeeper_team_0_velp_rel_robot"], 
                         info["goalkeeper_team_1_xpos_rel_robot"], 
                         info["goalkeeper_team_1_velp_rel_robot"], 
                         info["target_xpos_rel_robot"], 
                         info["target_velp_rel_robot"], 
                         info["defender_xpos"],
                         task_onehot))

        return obs

## Create the model
if args.alg == 'ddpg':
# Use single_action_space (not vectorized) for model creation
    model = DDPG_FF(
        n_features=89,  # type: ignore
        action_space=single_action_space,  # type: ignore - use single env action space
        neurons=[24, 12, 6],
        activation_function=F.relu,
        learning_rate=0.0001,
    )
elif args.alg == 'ppo':
    model = PPO(
        n_features=89,
        action_space=single_action_space,
        neurons=[24, 12, 6],
        activation_function=F.relu,
        learning_rate=0.0001,
    )

## Define an action function
def action_function(policy):
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (
        expected_bounds[1] - expected_bounds[0]
    )
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    
    # Use env.action_space like the original code - this works because env is in global scope
    # Handle both single and vectorized environments
    action_space = env.action_space
    if hasattr(action_space, 'single_action_space'):
        # Vectorized env - use single action space for broadcasting
        action_low = action_space.single_action_space.low
        action_high = action_space.single_action_space.high
    else:
        # Single env
        action_low = action_space.low
        action_high = action_space.high
    
    return (
        action_low
        + (action_high - action_low) * bounded_percent
    )

## Train the model
# Use PPO's own training_loop for on-policy learning
if isinstance(model, PPO):
    model.training_loop(env, action_function, Preprocessor, timesteps=args.timesteps)
else:
    # Use shared training_loop for off-policy algorithms (DDPG)
    training_loop(env, model, action_function, Preprocessor, timesteps=args.timesteps)

model.switch_to_test_mode()
# model.eval()
model.to("cpu")
## Watch
sai.watch(model, action_function, Preprocessor)

## Benchmark the model locally
sai.benchmark(model, action_function, Preprocessor)
