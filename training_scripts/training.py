from typing import Optional, Callable
import random
from collections import deque
import sys
import os

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

# Use gymnasium's built-in vector environments
try:
    from gymnasium.vector import SyncVectorEnv
except ImportError:
    # Fallback to old gym if gymnasium.vector is not available
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'diffusion_policy', 'diffusion_policy', 'gym_util'))
    from sync_vector_env import SyncVectorEnv


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


def add_noise(action, noise_scale=0.1):
    noise = np.random.normal(0, noise_scale, size=action.shape)
    return np.clip(action + noise, -1, 1)


def _is_vectorized(env):
    """Check if environment is vectorized (has num_envs attribute)"""
    return hasattr(env, 'num_envs') and env.num_envs > 1


def training_loop(
    env: gym.Env,
    model,
    action_function: Optional[Callable] = None,
    preprocess_class: Optional[Callable] = None,
    timesteps=1000,
):
    replay_buffer = ReplayBuffer(max_size=100000)
    preprocessor = preprocess_class()
    batch_size = 64
    update_frequency = 4

    is_vectorized = _is_vectorized(env)
    num_envs = env.num_envs if is_vectorized else 1
    print(f"Number of environments: {num_envs}")

    total_steps = 0
    episode_count = 0
    episode_rewards = [0.0] * num_envs
    episode_steps_list = [0] * num_envs

    pbar = tqdm(total=timesteps, desc="Training Progress", unit="steps")

    # Reset all environments
    if is_vectorized:
        # Gymnasium SyncVectorEnv returns (observations, infos) tuple
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            observations, infos = reset_result
        else:
            # Old gym format - just observations
            observations = reset_result
            infos = [{}] * num_envs
    else:
        observations, info = env.reset()
        observations = np.expand_dims(observations, axis=0)
        infos = [info]
    
    # Store current infos for use after resets
    current_infos = infos.copy() if is_vectorized else infos
    
    # Preprocess initial states
    if is_vectorized:
        # For vectorized, we need to handle batched info
        states = preprocessor.modify_state(observations, infos)
    else:
        states = preprocessor.modify_state(observations, infos[0])
        states = np.expand_dims(states, axis=0)

    dones = np.zeros(num_envs, dtype=bool)

    while total_steps < timesteps:
        # Get actions for all environments
        states_tensor = torch.from_numpy(states).float()
        policies = model(states_tensor).detach().cpu().numpy()
        
        # Ensure policies have correct shape: (num_envs, action_dim)
        # Handle cases where model outputs extra dimensions like (1, 1, 12) -> (1, 12)
        original_shape = policies.shape
        if len(policies.shape) > 2:
            # Reshape to (batch, action_dim) by flattening middle dimensions
            policies = policies.reshape(policies.shape[0], -1)
        elif len(policies.shape) == 1:
            # Add batch dimension: (action_dim,) -> (1, action_dim)
            policies = np.expand_dims(policies, axis=0)

        if action_function:
            actions = action_function(policies)
            actions = add_noise(actions, noise_scale=0.1)
        else:
            actions = np.array([model.select_action(s) for s in states])
        # Step all environments
        if is_vectorized:
            # Gymnasium SyncVectorEnv uses step() directly (not async)
            step_result = env.step(actions)
            # Gymnasium format: (obs, rewards, terminated, truncated, infos)
            if len(step_result) == 5:
                next_observations, rewards, terminated, truncated, next_infos = step_result
            else:
                # Old gym format: (obs, rewards, dones, infos)
                next_observations, rewards, dones, next_infos = step_result
                terminated = dones
                truncated = np.zeros_like(dones)
            dones = terminated | truncated
        else:
            # For single env, extract action and ensure it's 1D: (action_dim,)
            action = actions[0] if actions.shape[0] == 1 else actions
            step_result = env.step(action)
            if len(step_result) == 5:
                next_observations, rewards, terminated, truncated, next_info = step_result
            else:
                # Old gym format
                next_observations, rewards, done, next_info = step_result
                terminated = np.array([done])
                truncated = np.array([False])
            next_observations = np.expand_dims(next_observations, axis=0)
            rewards = np.array([rewards])
            terminated = np.array([terminated]) if not isinstance(terminated, np.ndarray) else terminated
            truncated = np.array([truncated]) if not isinstance(truncated, np.ndarray) else truncated
            next_infos = [next_info]

        # Preprocess next states
        if is_vectorized:
            next_states = preprocessor.modify_state(next_observations, next_infos)
            # Update current infos for use after resets
            current_infos = next_infos
        else:
            next_states = preprocessor.modify_state(next_observations, next_infos[0])
            next_states = np.expand_dims(next_states, axis=0)
            current_infos = next_infos

        dones = terminated | truncated

        # Add experiences to replay buffer
        for i in range(num_envs):
            replay_buffer.add(
                states[i].squeeze(),
                actions[i].squeeze(),
                rewards[i],
                next_states[i].squeeze(),
                dones[i]
            )
            episode_rewards[i] += rewards[i]
            episode_steps_list[i] += 1
            total_steps += 1
            pbar.update(1)

        # Reset done environments
        if is_vectorized:
            reset_mask = dones
            if reset_mask.any():
                # Count episodes for done environments
                reset_indices = np.where(reset_mask)[0]
                for idx in reset_indices:
                    episode_count += 1
                    episode_rewards[idx] = 0.0
                    episode_steps_list[idx] = 0
                
                # Reset all environments (SyncVectorEnv doesn't support selective reset)
                # This is simpler and correct, though slightly less efficient
                reset_result = env.reset()
                # Gymnasium SyncVectorEnv returns (observations, infos) tuple
                if isinstance(reset_result, tuple):
                    reset_obs, reset_infos = reset_result
                else:
                    # Old gym format - just observations
                    reset_obs = reset_result
                    reset_infos = [{}] * num_envs
                
                # Update states with reset observations
                reset_states = preprocessor.modify_state(reset_obs, reset_infos)
                states = reset_states
            else:
                states = next_states
        else:
            if dones[0]:
                episode_count += 1
                episode_reward = episode_rewards[0]
                episode_rewards[0] = 0.0
                episode_steps_list[0] = 0
                
                # Reset single environment
                observations, info = env.reset()
                observations = np.expand_dims(observations, axis=0)
                infos = [info]
                states = preprocessor.modify_state(observations, infos[0])
                states = np.expand_dims(states, axis=0)
            else:
                states = next_states

        # Update model
        if len(replay_buffer) >= batch_size and total_steps % update_frequency == 0:
            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.sample(
                batch_size
            )
            critic_loss, actor_loss = model.train(
                states_batch,
                actions_batch,
                rewards_batch.reshape(-1, 1),
                next_states_batch,
                dones_batch.reshape(-1, 1),
                1,
            )

            avg_reward = np.mean(episode_rewards) if is_vectorized else episode_rewards[0]
            pbar.set_description(
                f"Episode {episode_count} | Avg Reward: {avg_reward:.2f} | Critic: {critic_loss:.4f} | Actor: {actor_loss:.4f}"
            )

    pbar.close()
    env.close()
