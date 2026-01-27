"""
FPO (Flow Matching Policy Optimization) implementation.
Compatible with DDPG interface (forward, select_action, switch_to_test_mode).
Uses DiffusionPolicy as actor and NeuralNetwork as critic.
"""

import time
from dataclasses import dataclass
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from network import NeuralNetwork, DiffusionPolicy

# Optional wandb import
try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class FpoActionInfo:
    """
    Container for additional information returned by the FPO actor.
    """
    x_t_path: torch.Tensor         # (*, flow_steps, action_dim)
    loss_eps: torch.Tensor         # (*, sample_dim, action_dim)
    loss_t: torch.Tensor           # (*, sample_dim, 1)
    initial_cfm_loss: torch.Tensor # (*,)


class FPO(torch.nn.Module):
    """
    Flow Matching Policy Optimization (FPO).
    
    - Actor: DiffusionPolicy (flow-matching policy)
    - Critic: NeuralNetwork (standard value network)
    - Uses PPO-style clipped objective, where the policy ratio is derived
      from CFM loss differences instead of log-prob ratios.
    """
    
    def __init__(self, n_features, action_space, neurons, activation_function, learning_rate, **hyperparameters):
        """
        FPO agent with DiffusionPolicy actor and NeuralNetwork critic.
        Compatible with DDPG interface.
        
        Args:
            n_features: Observation dimension
            action_space: Action space (gymnasium Box)
            neurons: List of hidden layer sizes
            activation_function: Activation function (e.g., F.relu)
            learning_rate: Learning rate
            **hyperparameters: Additional FPO hyperparameters
        """
        super().__init__()
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Env info
        self.obs_dim = n_features
        self.act_dim = action_space.shape[0]
        
        # Initialize hyperparameters
        self._init_hyperparameters(hyperparameters)
        
        # Actor input dimension: observation + action + time
        actor_input_dim = n_features + self.act_dim + 1
        
        # Actor: DiffusionPolicy
        actor_kwargs = hyperparameters.get("actor_kwargs", {})
        self.actor = DiffusionPolicy(
            n_features=actor_input_dim,  # state + action + time
            n_actions=self.act_dim,       # output is velocity (same dim as action)
            neurons=neurons,
            activation_function=activation_function,
            state_dim=n_features,
            action_dim=self.act_dim,
            device=self.device,
            num_steps=hyperparameters.get("num_steps", 10),
            fixed_noise_inference=hyperparameters.get("fixed_noise_inference", False),
            **actor_kwargs,
        ).to(self.device)
        
        # Critic: NeuralNetwork
        self.critic = NeuralNetwork(
            n_features,
            1,  # value output
            neurons,
            activation_function,
        ).to(self.device)
        
        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # FPO-specific hyperparameters
        self.num_fpo_samples = hyperparameters.get("num_fpo_samples", 100)
        self.positive_advantage = hyperparameters.get("positive_advantage", False)
        
        print(f"[FPO] Actor: DiffusionPolicy with {actor_input_dim} input dim")
        print(f"[FPO] Critic: NeuralNetwork with {n_features} input dim")
        print(f"[FPO] Training with {self.num_fpo_samples} CFM samples per state")
        print(f"[FPO] positive_advantage = {self.positive_advantage}")
        
        # Logger
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,
            "i_so_far": 0,
            "batch_lens": [],
            "batch_rews": [],
            "actor_losses": [],
        }

    # ---------------------------------------------------------------------- #
    #                        DDPG-compatible interface                       #
    # ---------------------------------------------------------------------- #
    def switch_to_test_mode(self):
        """Move model to CPU for inference."""
        self.device = torch.device("cpu")
        # Use eval() to ensure no gradient computation
        self.eval()
        # Move all model parameters and buffers to CPU
        self.to("cpu")
        # Explicitly move submodules to ensure they're on CPU
        self.actor.eval()
        self.actor.to("cpu")
        # Update DiffusionPolicy's device attribute and move init_noise buffer
        if hasattr(self.actor, 'device'):
            self.actor.device = torch.device("cpu")
        if hasattr(self.actor, 'init_noise'):
            self.actor.init_noise = self.actor.init_noise.to("cpu")
        
        self.critic.eval()
        self.critic.to("cpu")
        
        # Ensure all parameters and buffers are on CPU (double-check)
        for param in self.parameters():
            if param.is_cuda:
                param.data = param.data.cpu()
        for buffer in self.buffers():
            if buffer.is_cuda:
                buffer.data = buffer.data.cpu()
        # Also check actor and critic parameters/buffers
        for param in self.actor.parameters():
            if param.is_cuda:
                param.data = param.data.cpu()
        for buffer in self.actor.buffers():
            if buffer.is_cuda:
                buffer.data = buffer.data.cpu()
        for param in self.critic.parameters():
            if param.is_cuda:
                param.data = param.data.cpu()
        for buffer in self.critic.buffers():
            if buffer.is_cuda:
                buffer.data = buffer.data.cpu()

    def forward(self, current_layer):
        """
        Forward pass: returns deterministic action (mean of diffusion policy).
        Compatible with DDPG interface.
        """
        if isinstance(current_layer, np.ndarray):
            current_layer = torch.from_numpy(current_layer)
        
        # Handle shape: ensure it's 2D (batch, features)
        if len(current_layer.shape) > 2:
            current_layer = current_layer.reshape(current_layer.shape[0], -1)
        elif len(current_layer.shape) == 1:
            current_layer = current_layer.unsqueeze(0)
        
        # Move input to model's device (should be CPU after switch_to_test_mode)
        current_layer = current_layer.float().to(self.device)
        # DiffusionPolicy's sample_action returns deterministic action
        action = self.actor.sample_action(current_layer)
        action = action.reshape(1, -1)
        return action

    def select_action(self, state):
        """
        Sample an action from the FPO policy (deterministic).
        Compatible with DDPG interface.
        """
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().to(self.device)
        else:
            state_tensor = state.float().to(self.device)
        
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor.sample_action(state_tensor)
        return action.detach().cpu().numpy().squeeze()

    def get_action_with_info(self, obs):
        """
        Get action and FpoActionInfo (used internally for on-policy collection).
        """
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
        else:
            obs_tensor = obs.float().to(self.device)
        
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        with torch.no_grad():
            action, x_t_path, eps, t, initial_cfm_loss = self.actor.sample_action_with_info(
                obs_tensor,
                num_train_samples=self.num_fpo_samples,
            )
        
        action_info = FpoActionInfo(
            x_t_path=x_t_path,
            loss_eps=eps,
            loss_t=t,
            initial_cfm_loss=initial_cfm_loss,
        )
        
        action_np = action.detach().cpu().numpy()
        if len(action_np.shape) > 1:
            action_np = action_np.squeeze()
        
        return action_np, action_info

    # ---------------------------------------------------------------------- #
    #                        FPO Training Loop                               #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _is_vectorized(env):
        """Check if environment is vectorized (has num_envs attribute)"""
        return hasattr(env, 'num_envs') and env.num_envs > 1

    def training_loop(self, env, action_function=None, preprocess_class=None, timesteps=1000):
        """
        FPO training loop compatible with DDPG interface.
        
        Args:
            env: Gymnasium environment (single or vectorized)
            action_function: Optional action transformation function (not used for FPO)
            preprocess_class: Preprocessor class for state preprocessing
            timesteps: Total number of timesteps to train
        """
        from tqdm import tqdm
        
        preprocessor = preprocess_class() if preprocess_class else None
        is_vectorized = self._is_vectorized(env)
        num_envs = env.num_envs if is_vectorized else 1
        
        print(f"FPO Training: {num_envs} environment(s), {timesteps} timesteps")
        print(f"Batch size: {self.timesteps_per_batch}, Max episode length: {self.max_timesteps_per_episode}")
        
        total_steps = 0
        iteration = 0
        pbar = tqdm(total=timesteps, desc="FPO Training", unit="steps")
        
        while total_steps < timesteps:
            # ------------------------------------------------------------------
            # 1. Rollout: collect on-policy data with CFM info
            # ------------------------------------------------------------------
            (
                batch_obs,
                batch_acts,
                batch_action_info,
                batch_rews,
                batch_lens,
                batch_vals,
                batch_dones,
            ) = self._rollout_with_preprocessor(env, preprocessor, is_vectorized, num_envs)
            
            if batch_obs.size(0) == 0:
                continue
                
            batch_obs = batch_obs.to(self.device)
            batch_acts = batch_acts.to(self.device)
            
            # ------------------------------------------------------------------
            # 2. GAE Advantage + target values
            # ------------------------------------------------------------------
            # Compute values for all observations first
            with torch.no_grad():
                values = self.critic(batch_obs).squeeze()
            
            # Flatten episode data to match batch_obs structure
            flat_rews = []
            flat_vals = []
            flat_dones = []

            for ep_rews, ep_vals, ep_dones in zip(batch_rews, batch_vals, batch_dones):
                flat_rews.extend(ep_rews)
                flat_vals.extend(ep_vals)
                flat_dones.extend(ep_dones)
            
            # Convert to tensors
            flat_rews = torch.tensor(flat_rews, dtype=torch.float32).to(self.device)
            flat_vals_tensor = torch.tensor(flat_vals, dtype=torch.float32).to(self.device)
            flat_dones = torch.tensor(flat_dones, dtype=torch.float32).to(self.device)
            
            # Compute GAE advantages
            advantages = torch.zeros_like(values)
            last_advantage = 0.0
            
            # Process backwards through the flattened data
            for t in reversed(range(len(flat_rews))):
                if t + 1 < len(flat_rews):
                    delta = (
                        flat_rews[t]
                        + self.gamma * flat_vals_tensor[t + 1] * (1 - flat_dones[t + 1])
                        - flat_vals_tensor[t]
                    )
                else:
                    delta = flat_rews[t] - flat_vals_tensor[t]
                
                # GAE: A_t = δ_t + (γλ)(1 - done_t) * A_{t+1}
                advantages[t] = delta + self.gamma * self.lam * (1 - flat_dones[t]) * last_advantage
                last_advantage = advantages[t]
            
            batch_rtgs = advantages + values  # target values (V-target)
            
            # Update counters
            total_steps += np.sum(batch_lens)
            iteration += 1
            self.logger["t_so_far"] = total_steps
            self.logger["i_so_far"] = iteration
            
            # Normalize advantages
            if self.positive_advantage:
                advantages = F.softplus(advantages)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            
            # ------------------------------------------------------------------
            # 3. FPO update (mini-batch SGD for several epochs)
            # ------------------------------------------------------------------
            num_steps = batch_obs.size(0)
            indices = np.arange(num_steps)
            minibatch_size = max(num_steps // self.num_minibatches, 1)
            epoch_actor_losses = []
            
            for _ in range(self.n_updates_per_iteration):
                # Learning rate annealing
                frac = (total_steps - 1.0) / timesteps
                new_lr = max(self.learning_rate * (1.0 - frac), 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                self.logger["lr"] = new_lr
                
                np.random.shuffle(indices)
                
                for start in range(0, num_steps, minibatch_size):
                    end = start + minibatch_size
                    mb_idx = indices[start:end]
                    if len(mb_idx) == 0:
                        continue
                    
                    mb_obs = batch_obs[mb_idx]
                    mb_acts = batch_acts[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_rtgs = batch_rtgs[mb_idx]
                    mb_infos = [batch_action_info[i] for i in mb_idx]
                    
                    # Extract CFM-related tensors
                    loss_eps = torch.stack([info.loss_eps for info in mb_infos]).to(self.device)  # [B, N, act_dim]
                    loss_t = torch.stack([info.loss_t for info in mb_infos]).to(self.device)        # [B, N, 1]
                    initial_cfm_loss = torch.stack([info.initial_cfm_loss for info in mb_infos]).to(self.device)  # [B, N]
                    
                    # Critic prediction
                    V_pred = self.critic(mb_obs).squeeze()
                    
                    # Compute CFM loss difference
                    B, N, D = loss_eps.shape
                    
                    # Repeat observations and actions across N samples
                    flat_obs = mb_obs.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)   # [B*N, obs_dim]
                    flat_acts = mb_acts.unsqueeze(1).repeat(1, N, 1).view(B * N, -1) # [B*N, act_dim]
                    
                    # Flatten eps, t, and initial_cfm_loss
                    flat_eps = loss_eps.view(B * N, -1)                                 # [B*N, act_dim]
                    flat_t = loss_t.view(B * N, -1)                                     # [B*N, 1]
                    flat_init_loss = initial_cfm_loss.view(B * N)                       # [B*N]
                    
                    # Compute the new CFM loss with current actor
                    # DiffusionPolicy.compute_cfm_loss expects: (state_norm, x1, eps, t)
                    # where state_norm is the observation, x1 is the action, eps is noise, t is time
                    cfm_loss = self.actor.compute_cfm_loss(
                        flat_obs,  # state_norm: [B*N, obs_dim]
                        flat_acts, # x1: [B*N, act_dim] - final action
                        flat_eps,  # eps: [B*N, act_dim] - noise
                        flat_t,    # t: [B*N, 1] - time
                    )  # [B*N]
                    
                    # Compute per-sample CFM loss difference and reshape to [B, N]
                    diff = flat_init_loss - cfm_loss
                    cfm_difference = diff.view(B, N)
                    
                    # Build state-wise policy ratio rho_s from CFM diff
                    cfm_difference = torch.clamp(cfm_difference, -3.0, 3.0)
                    delta_s = cfm_difference.mean(dim=1)  # [B]
                    delta_s = torch.clamp(delta_s, -3.0, 3.0)
                    rho_s = torch.exp(delta_s)  # [B]
                    
                    # PPO-style clipped surrogate objective with rho_s
                    surr1 = rho_s * mb_adv
                    surr2 = torch.clamp(rho_s, 1 - self.clip, 1 + self.clip) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Actor loss (no entropy for FPO, but keep structure)
                    entropy = torch.tensor(0.0, device=self.device)
                    actor_loss = policy_loss - self.ent_coef * entropy.mean()
                    
                    # Critic loss
                    critic_loss = nn.MSELoss()(V_pred, mb_rtgs)
                    
                    # Backward: actor
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()
                    
                    # Backward: critic
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()
                    
                    epoch_actor_losses.append(actor_loss.detach())
                    
                    # Logging
                    if wandb is not None:
                        wandb.log({
                            "cfm_difference": cfm_difference.mean().item(),
                            "policy_ratio_mean": rho_s.mean().item(),
                            "policy_loss": policy_loss.item(),
                            "actor_loss": actor_loss.item(),
                            "critic_loss": critic_loss.item(),
                        })
            
            # ------------------------------------------------------------------
            # 4. Logging
            # ------------------------------------------------------------------
            avg_actor_loss = sum(epoch_actor_losses) / len(epoch_actor_losses) if epoch_actor_losses else 0.0
            self.logger["actor_losses"].append(avg_actor_loss)
            self.logger["batch_rews"] = batch_rews
            self.logger["batch_lens"] = batch_lens
            
            avg_ep_ret = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
            avg_ep_len = np.mean(batch_lens)
            pbar.update(np.sum(batch_lens))
            pbar.set_description(
                f"Iter {iteration} | Steps: {total_steps}/{timesteps} | "
                f"Return: {avg_ep_ret:.2f} | Len: {avg_ep_len:.1f} | "
                f"Loss: {avg_actor_loss:.4f}"
            )
            
            if wandb is not None:
                wandb.log({
                    "advantage_hist": wandb.Histogram(advantages.cpu().numpy()),
                    "actor_loss": avg_actor_loss,
                    "Return": avg_ep_ret,
                })
        
        pbar.close()
        env.close()
        print(f"FPO Training completed: {total_steps} timesteps, {iteration} iterations")

    def _rollout_with_preprocessor(self, env, preprocessor, is_vectorized, num_envs):
        """
        Collect one batch of on-policy data with preprocessor support.
        Handles both single and vectorized environments.
        Returns action_info for FPO.
        """
        batch_obs = []
        batch_acts = []
        batch_action_info = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []
        
        t = 0
        
        # Track episodes per environment for vectorized case
        if is_vectorized:
            env_obs = [None] * num_envs
            env_dones = [False] * num_envs
            env_ep_rews = [[] for _ in range(num_envs)]
            env_ep_vals = [[] for _ in range(num_envs)]
            env_ep_dones = [[] for _ in range(num_envs)]
            env_ep_lens = [0] * num_envs
            
            # Reset all environments
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                observations, infos = reset_result
            else:
                observations = reset_result
                infos = [{}] * num_envs
            
            # Preprocess initial states
            if preprocessor:
                states = preprocessor.modify_state(observations, infos)
            else:
                states = observations
            
            for i in range(num_envs):
                env_obs[i] = states[i]
        
        while t < self.timesteps_per_batch:
            if is_vectorized:
                # Vectorized: collect from all environments in parallel
                actions = []
                step_data = []
                
                for i in range(num_envs):
                    if env_dones[i]:
                        actions.append(np.zeros(self.act_dim))
                    else:
                        obs = env_obs[i]
                        action, action_info = self.get_action_with_info(obs)
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                        if len(obs_tensor.shape) == 1:
                            obs_tensor = obs_tensor.unsqueeze(0)
                        value = self.critic(obs_tensor).detach().squeeze()
                        
                        actions.append(action)
                        step_data.append((i, obs, action, action_info, value.item()))
                
                # Step all environments
                actions_array = np.array(actions)
                step_result = env.step(actions_array)
                if len(step_result) == 5:
                    next_observations, rewards, terminated, truncated, next_infos = step_result
                else:
                    next_observations, rewards, dones, next_infos = step_result
                    terminated = dones
                    truncated = np.zeros_like(dones)
                
                dones = terminated | truncated
                
                # Preprocess next states
                if preprocessor:
                    next_states = preprocessor.modify_state(next_observations, next_infos)
                else:
                    next_states = next_observations
                
                # Process step data
                step_idx = 0
                for i in range(num_envs):
                    if env_dones[i]:
                        env_obs[i] = next_states[i]
                        env_dones[i] = False
                        env_ep_rews[i] = []
                        env_ep_vals[i] = []
                        env_ep_dones[i] = []
                        env_ep_lens[i] = 0
                    else:
                        env_idx, obs, action, action_info, value = step_data[step_idx]
                        rew = rewards[i]
                        done = dones[i]
                        
                        batch_obs.append(obs)
                        batch_acts.append(action)
                        batch_action_info.append(action_info)
                        env_ep_rews[i].append(rew)
                        env_ep_vals[i].append(value)
                        env_ep_dones[i].append(env_dones[i])
                        env_ep_lens[i] += 1
                        t += 1
                        step_idx += 1
                        
                        if done:
                            env_dones[i] = True
                            batch_lens.append(env_ep_lens[i])
                            batch_rews.append(env_ep_rews[i])
                            batch_vals.append(env_ep_vals[i])
                            batch_dones.append(env_ep_dones[i])
                            
                            env_obs[i] = next_states[i]
                            env_dones[i] = False
                            env_ep_rews[i] = []
                            env_ep_vals[i] = []
                            env_ep_dones[i] = []
                            env_ep_lens[i] = 0
                        else:
                            env_obs[i] = next_states[i]
            else:
                # Single environment rollout
                ep_rews = []
                ep_vals = []
                ep_dones = []
                
                # Reset environment
                observations, info = env.reset()
                if preprocessor:
                    obs = preprocessor.modify_state(
                        np.expand_dims(observations, axis=0),
                        info
                    )
                    obs = obs.squeeze() if len(obs.shape) > 1 else obs[0]
                else:
                    obs = observations
                
                done = False
                
                for ep_t in range(self.max_timesteps_per_episode):
                    ep_dones.append(done)
                    
                    t += 1
                    batch_obs.append(obs)
                    
                    # Get action and action_info
                    action, action_info = self.get_action_with_info(obs)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    value = self.critic(obs_tensor).detach().squeeze()
                    
                    # Step environment
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_obs_raw, rew, terminated_flag, truncated_flag, next_info = step_result
                    else:
                        next_obs_raw, rew, done_flag, next_info = step_result
                        terminated_flag = done_flag
                        truncated_flag = False
                    
                    done = terminated_flag or truncated_flag
                    
                    # Preprocess next observation
                    if preprocessor:
                        next_obs = preprocessor.modify_state(
                            np.expand_dims(next_obs_raw, axis=0),
                            next_info
                        )
                        next_obs = next_obs.squeeze() if len(next_obs.shape) > 1 else next_obs[0]
                    else:
                        next_obs = next_obs_raw
                    
                    ep_rews.append(rew)
                    ep_vals.append(value.item())
                    batch_acts.append(action)
                    batch_action_info.append(action_info)
                    
                    obs = next_obs
                    
                    if done:
                        break
                
                batch_lens.append(ep_t + 1)
                batch_rews.append(ep_rews)
                batch_vals.append(ep_vals)
                batch_dones.append(ep_dones)
        
        # Handle ongoing episodes in vectorized case
        if is_vectorized:
            for i in range(num_envs):
                if not env_dones[i] and len(env_ep_rews[i]) > 0:
                    batch_lens.append(env_ep_lens[i])
                    batch_rews.append(env_ep_rews[i])
                    batch_vals.append(env_ep_vals[i])
                    batch_dones.append(env_ep_dones[i])
        
        if len(batch_obs) == 0:
            return (
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.float32),
                [],
                [],
                [],
                [],
                [],
            )
        
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32)
        
        return (
            batch_obs,
            batch_acts,
            batch_action_info,
            batch_rews,
            batch_lens,
            batch_vals,
            batch_dones,
        )

    # ---------------------------------------------------------------------- #
    #                        Hyperparameters & Logging                       #
    # ---------------------------------------------------------------------- #

    def _init_hyperparameters(self, hyperparameters):
        """
        Set default hyperparameters and override with user-provided values.
        """
        # Core FPO hyperparameters (similar to PPO)
        self.timesteps_per_batch = 2000
        self.max_timesteps_per_episode = 600
        self.n_updates_per_iteration = 1
        self.gamma = 0.95
        self.clip = 0.2

        # Extras
        self.lam = 0.98
        self.num_minibatches = 6
        self.ent_coef = 0.0
        self.max_grad_norm = 0.5

        # Misc
        self.render = False
        self.render_every_i = 10
        self.save_freq = 10
        self.seed = None
        self.run_name = "fpo_run"

        # Override defaults
        for param, val in hyperparameters.items():
            setattr(self, param, val)

        if self.seed is not None:
            assert isinstance(self.seed, int)
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")
