"""
PPO implementation with dedicated training loop for on-policy learning.
Compatible with DDPG interface (forward, select_action, switch_to_test_mode).
"""

import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from network import NeuralNetwork

# Optional wandb import
try:
    import wandb
except ImportError:
    wandb = None


class PPO(torch.nn.Module):
    def __init__(self, n_features, action_space, neurons, activation_function, learning_rate, **hyperparameters):
        """
        PPO agent with separate actor / critic networks.
        Compatible with DDPG interface.
        
        Args:
            n_features: Observation dimension
            action_space: Action space (gymnasium Box)
            neurons: List of hidden layer sizes
            activation_function: Activation function (e.g., F.relu)
            learning_rate: Learning rate
            **hyperparameters: Additional PPO hyperparameters
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

        # Actor / Critic networks (using NeuralNetwork like DDPG)
        shared_inputs = [neurons, activation_function]
        self.actor = NeuralNetwork(
            n_features,
            action_space.shape[0],
            *shared_inputs,
            F.tanh,  # Tanh for bounded actions
        ).to(self.device)
        self.critic = NeuralNetwork(
            n_features, 1, *shared_inputs
        ).to(self.device)

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

        # Action distribution covariance
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        
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
        self.device = torch.device("cpu")


    def forward(self, current_layer):
        """
        Forward pass: returns mean action (deterministic).
        Compatible with DDPG interface.
        """
        if isinstance(current_layer, np.ndarray):
            current_layer = torch.from_numpy(current_layer)
        
        # Handle shape: ensure it's 2D (batch, features)
        if len(current_layer.shape) > 2:
            # Reshape from (1, 1, features) or similar to (1, features)
            current_layer = current_layer.reshape(current_layer.shape[0], -1)
        elif len(current_layer.shape) == 1:
            # Add batch dimension: (features,) -> (1, features)
            current_layer = current_layer.unsqueeze(0)
        
        current_layer = current_layer.float().to(self.device)
        return self.actor.forward(current_layer)

    def select_action(self, state):
        """
        Sample an action from the current policy π(a|s).
        Compatible with DDPG interface but returns action only (log_prob stored separately).
        """
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().to(self.device)
        else:
            state_tensor = state.float().to(self.device)
        mean = self.actor(state_tensor)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        return action.detach().cpu().numpy()

    def get_action_with_log_prob(self, obs):
        """
        Get action and log_prob (used internally for on-policy collection).
        Returns log_prob as a numpy scalar for storage in replay buffer.
        """
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
        else:
            obs_tensor = obs.float().to(self.device)
        
        # Handle shape: ensure it's 2D (batch, features)
        if len(obs_tensor.shape) > 2:
            obs_tensor = obs_tensor.reshape(obs_tensor.shape[0], -1)
        elif len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        mean = self.actor(obs_tensor)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return action and log_prob as numpy arrays
        action_np = action.detach().cpu().numpy()
        log_prob_np = log_prob.detach().cpu().numpy()
        
        # If single sample, squeeze to 1D action and scalar log_prob
        if len(action_np.shape) > 1:
            action_np = action_np.squeeze()
        # Ensure log_prob is a scalar (float) for storage
        if log_prob_np.size == 1:
            log_prob_np = float(log_prob_np.item())
        else:
            log_prob_np = log_prob_np.squeeze()
            if log_prob_np.size == 1:
                log_prob_np = float(log_prob_np.item())
        
        return action_np, log_prob_np

    # ---------------------------------------------------------------------- #
    #                        PPO Training Loop                               #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _is_vectorized(env):
        """Check if environment is vectorized (has num_envs attribute)"""
        return hasattr(env, 'num_envs') and env.num_envs > 1

    def training_loop(self, env, action_function=None, preprocess_class=None, timesteps=1000):
        """
        PPO training loop compatible with DDPG interface.
        
        Args:
            env: Gymnasium environment (single or vectorized)
            action_function: Optional action transformation function (not used for PPO)
            preprocess_class: Preprocessor class for state preprocessing
            timesteps: Total number of timesteps to train
        """
        import gymnasium as gym
        from tqdm import tqdm
        
        preprocessor = preprocess_class() if preprocess_class else None
        is_vectorized = self._is_vectorized(env)
        num_envs = env.num_envs if is_vectorized else 1
        
        print(f"PPO Training: {num_envs} environment(s), {timesteps} timesteps")
        print(f"Batch size: {self.timesteps_per_batch}, Max episode length: {self.max_timesteps_per_episode}")
        
        total_steps = 0
        iteration = 0
        pbar = tqdm(total=timesteps, desc="PPO Training", unit="steps")
        
        while total_steps < timesteps:
            # ------------------------------------------------------------------
            # 1. Rollout: collect on-policy data
            # ------------------------------------------------------------------
            (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rews,
                batch_lens,
                batch_vals,
                batch_dones,
            ) = self._rollout_with_preprocessor(env, preprocessor, is_vectorized, num_envs)
            
            if batch_obs.size(0) == 0:
                continue
                
            batch_obs = batch_obs.to(self.device)
            batch_acts = batch_acts.to(self.device)
            batch_log_probs = batch_log_probs.to(self.device)
            
            # ------------------------------------------------------------------
            # 2. GAE Advantage + target values
            # ------------------------------------------------------------------
            # Compute values for all observations first
            with torch.no_grad():
                values = self.critic(batch_obs).squeeze()
            
            # Flatten episode data to match batch_obs structure
            # batch_obs contains observations in order, so we need to match that
            flat_rews = []
            flat_vals = []
            flat_dones = []
            
            for ep_rews, ep_vals, ep_dones in zip(batch_rews, batch_vals, batch_dones):
                flat_rews.extend(ep_rews)
                flat_vals.extend(ep_vals)
                flat_dones.extend(ep_dones)
            
            # Check if sizes match - if not, trim batch_obs to match episode data
            # This can happen if we collected observations beyond episode boundaries
            if len(flat_rews) != batch_obs.size(0):
                # Trim batch_obs, batch_acts, batch_log_probs to match episode data length
                expected_len = len(flat_rews)
                if expected_len < batch_obs.size(0):
                    batch_obs = batch_obs[:expected_len]
                    batch_acts = batch_acts[:expected_len]
                    batch_log_probs = batch_log_probs[:expected_len]
                    # Recompute values for trimmed observations
                    with torch.no_grad():
                        values = self.critic(batch_obs).squeeze()
                else:
                    # This shouldn't happen, but handle it
                    print(f"Warning: More episode data ({expected_len}) than observations ({batch_obs.size(0)})")
                    # Pad with zeros (not ideal, but prevents crash)
                    pad_len = expected_len - batch_obs.size(0)
                    flat_rews = flat_rews[:batch_obs.size(0)]
                    flat_vals = flat_vals[:batch_obs.size(0)]
                    flat_dones = flat_dones[:batch_obs.size(0)]
            
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
                    # Not the last timestep
                    delta = (
                        flat_rews[t]
                        + self.gamma * flat_vals_tensor[t + 1] * (1 - flat_dones[t + 1])
                        - flat_vals_tensor[t]
                    )
                else:
                    # Last timestep
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
            
            # Normalize advantages (stabilizes training)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            
            # ------------------------------------------------------------------
            # 3. PPO update (mini-batch SGD for several epochs)
            # ------------------------------------------------------------------
            num_steps = batch_obs.size(0)
            indices = np.arange(num_steps)
            minibatch_size = max(num_steps // self.num_minibatches, 1)
            epoch_actor_losses = []
            approx_kl = 0.0
            
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
                    mb_log_probs_old = batch_log_probs[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_rtgs = batch_rtgs[mb_idx]
                    
                    # Forward pass: evaluate V(s), log π(a|s), entropy
                    V_pred, log_probs_new, entropy = self.evaluate(mb_obs, mb_acts)
                    
                    # PPO clipped surrogate objective
                    log_ratio = log_probs_new - mb_log_probs_old
                    ratio = torch.exp(log_ratio)
                    
                    # Compute approximate KL divergence for early stopping
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    
                    # Compute the clipped surrogate objective
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_adv
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Add entropy bonus (to encourage exploration)
                    actor_loss -= self.ent_coef * entropy.mean()
                    
                    # Compute value function loss (critic_loss) using MSE
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
                
                # Early stop if KL too large
                if approx_kl > self.target_kl:
                    break
            
            # ------------------------------------------------------------------
            # 4. Logging
            # ------------------------------------------------------------------
            avg_actor_loss = sum(epoch_actor_losses) / len(epoch_actor_losses) if epoch_actor_losses else 0.0
            self.logger["actor_losses"].append(avg_actor_loss)
            self.logger["batch_rews"] = batch_rews
            self.logger["batch_lens"] = batch_lens
            
            # Update progress bar
            avg_ep_ret = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
            avg_ep_len = np.mean(batch_lens)
            pbar.update(np.sum(batch_lens))
            pbar.set_description(
                f"Iter {iteration} | Steps: {total_steps}/{timesteps} | "
                f"Avg Return: {avg_ep_ret:.2f} | Avg Len: {avg_ep_len:.1f} | "
                f"Actor Loss: {avg_actor_loss:.4f}"
            )
            
            # if wandb is not None and iteration % 10 == 0:
            #     wandb.log({"advantage_hist": wandb.Histogram(advantages.cpu().numpy())})
        
        pbar.close()
        env.close()
        print(f"PPO Training completed: {total_steps} timesteps, {iteration} iterations")

    def _rollout_with_preprocessor(self, env, preprocessor, is_vectorized, num_envs):
        """
        Collect one batch of on-policy data with preprocessor support.
        Handles both single and vectorized environments.
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
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
                # Get actions for all environments
                actions = []
                step_data = []  # Store (env_idx, obs, action, log_prob, value) for non-done envs
                
                for i in range(num_envs):
                    if env_dones[i]:
                        # Use zero action for done envs (will be reset after step)
                        actions.append(np.zeros(self.act_dim))
                    else:
                        obs = env_obs[i]
                        action, log_prob = self.get_action_with_log_prob(obs)
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                        if len(obs_tensor.shape) == 1:
                            obs_tensor = obs_tensor.unsqueeze(0)
                        value = self.critic(obs_tensor).detach().squeeze()
                        
                        actions.append(action)
                        # Store step data - we'll add to batch after we get rewards
                        step_data.append((i, obs, action, log_prob, value.item()))
                
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
                
                # Process step data and update environments
                # Only process environments that actually took a step (non-done)
                step_idx = 0
                for i in range(num_envs):
                    if env_dones[i]:
                        # Done env was reset by vectorized env - skip processing
                        env_obs[i] = next_states[i]
                        env_dones[i] = False
                        env_ep_rews[i] = []
                        env_ep_vals[i] = []
                        env_ep_dones[i] = []
                        env_ep_lens[i] = 0
                    else:
                        # This environment took a step
                        env_idx, obs, action, log_prob, value = step_data[step_idx]
                        rew = rewards[i]
                        done = dones[i]
                        
                        # Store data for this step (observation BEFORE step, reward AFTER step)
                        batch_obs.append(obs)
                        batch_acts.append(action)
                        batch_log_probs.append(log_prob)
                        env_ep_rews[i].append(rew)
                        env_ep_vals[i].append(value)
                        env_ep_dones[i].append(env_dones[i])  # done flag BEFORE this step
                        env_ep_lens[i] += 1
                        t += 1
                        step_idx += 1
                        
                        if done:
                            env_dones[i] = True
                            # Episode ended - store episode data
                            batch_lens.append(env_ep_lens[i])
                            batch_rews.append(env_ep_rews[i])
                            batch_vals.append(env_ep_vals[i])
                            batch_dones.append(env_ep_dones[i])
                            
                            # Reset this environment
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
                    
                    # Get action and log_prob
                    action, log_prob = self.get_action_with_log_prob(obs)
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
                    batch_log_probs.append(log_prob)
                    
                    obs = next_obs
                    
                    if done:
                        break
                
                batch_lens.append(ep_t + 1)
                batch_rews.append(ep_rews)
                batch_vals.append(ep_vals)
                batch_dones.append(ep_dones)
        
        # Handle ongoing episodes in vectorized case - store their data even if not ended
        # This ensures we don't lose data from ongoing episodes at the batch boundary
        if is_vectorized:
            for i in range(num_envs):
                if not env_dones[i] and len(env_ep_rews[i]) > 0:
                    # Episode is still ongoing but has data - store it
                    batch_lens.append(env_ep_lens[i])
                    batch_rews.append(env_ep_rews[i])
                    batch_vals.append(env_ep_vals[i])
                    batch_dones.append(env_ep_dones[i])
        
        if len(batch_obs) == 0:
            return (
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.float32),
                [],
                [],
                [],
                [],
            )
        
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float32)
        
        return (
            batch_obs,
            batch_acts,
            batch_log_probs,
            batch_rews,
            batch_lens,
            batch_vals,
            batch_dones,
        )

    # ---------------------------------------------------------------------- #
    #                        Original PPO methods (kept for reference)       #
    # ---------------------------------------------------------------------- #
    #                          GAE Calculation                               #
    # ---------------------------------------------------------------------- #

    def calculate_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE) over a batch of episodes.
        """
        all_advantages = []

        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0.0

            # backward through episode
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = (
                        ep_rews[t]
                        + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1])
                        - ep_vals[t]
                    )
                else:
                    delta = ep_rews[t] - ep_vals[t]

                adv = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = adv
                advantages.insert(0, adv)

            all_advantages.extend(advantages)

        return torch.tensor(all_advantages, dtype=torch.float32)

    def evaluate(self, batch_obs, batch_acts):
        """
        Given a batch of (s, a), compute:
        - V(s)
        - log π(a|s)
        - entropy[π(·|s)]
        """
        values = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return values, log_probs, dist.entropy()

    # ---------------------------------------------------------------------- #
    #                        Hyperparameters & Logging                       #
    # ---------------------------------------------------------------------- #

    def _init_hyperparameters(self, hyperparameters):
        """
        Set default hyperparameters and override with user-provided values.
        """
        # Core PPO hyperparameters
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5
        self.gamma = 0.95
        self.clip = 0.2

        # Extras
        self.lam = 0.98
        self.num_minibatches = 6
        self.ent_coef = 0.0
        self.target_kl = 0.02
        self.max_grad_norm = 0.5
        self.deterministic = False

        # Misc
        self.render = True
        self.render_every_i = 10
        self.save_freq = 10
        self.seed = None
        self.run_name = "unnamed_run"

        # Override defaults
        for param, val in hyperparameters.items():
            setattr(self, param, val)

        if self.seed is not None:
            assert isinstance(self.seed, int)
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

