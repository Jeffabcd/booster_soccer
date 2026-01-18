from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class BoosterLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        dataset_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
    ):
        super().__init__()

        # Load the NPZ file
        data = np.load(dataset_path, allow_pickle=True)
        observations = data["observations"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        dones = data["done"]

        # Split into episodes based on done flags
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        
        episode_start = 0
        for i in range(len(dones)):
            if dones[i]:
                # Episode ends at index i (inclusive)
                episode_obs = observations[episode_start : i + 1]
                episode_actions = actions[episode_start : i + 1]
                
                # Only add episodes with at least 2 timesteps
                if len(episode_obs) > 1:
                    episode_data = {
                        "obs": episode_obs,
                        "action": episode_actions,
                    }
                    self.replay_buffer.add_episode(episode_data)
                
                episode_start = i + 1

        # Handle the last episode if it doesn't end with done=True
        if episode_start < len(dones):
            episode_obs = observations[episode_start:]
            episode_actions = actions[episode_start:]
            if len(episode_obs) > 1:
                episode_data = {
                    "obs": episode_obs,
                    "action": episode_actions,
                }
                self.replay_buffer.add_episode(episode_data)

        print(f"Loaded {self.replay_buffer.n_episodes} episodes from dataset")
        print(f"Observation dim: {observations.shape[1]}")
        print(f"Action dim: {actions.shape[1]}")

        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "obs": self.replay_buffer["obs"],
            "action": self.replay_buffer["action"],
        }
        if "range_eps" not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs["range_eps"] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
