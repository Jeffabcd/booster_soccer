import torch
import torch.nn as nn
import torch.nn.functional as F
from network import NeuralNetwork



class DDPG_FF(torch.nn.Module):
    def __init__(
        self, n_features, action_space, neurons, activation_function, learning_rate
    ):
        super().__init__()
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.tau = 0.001

        shared_inputs = [neurons, activation_function]
        self.actor = NeuralNetwork(
            n_features,
            action_space.shape[0],
            *shared_inputs,
            F.tanh,
        )
        self.critic = NeuralNetwork(
            n_features + action_space.shape[0], 1, *shared_inputs
        )

        self.target_actor = NeuralNetwork(
            n_features,
            action_space.shape[0],
            *shared_inputs,
            F.tanh,
        )
        self.target_critic = NeuralNetwork(
            n_features + action_space.shape[0], 1, *shared_inputs
        )

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.learning_rate
        )

    def soft_update_targets(self):
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    def switch_to_test_mode(self):
        self.device = torch.device("cpu")

    @staticmethod
    def backprop(optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
        optimizer.step()

    @staticmethod
    def get_critic_state(state, action):
        return torch.cat([state, action], dim=1)

    @staticmethod
    def tensor_to_array(torch_tensor):
        return torch_tensor.detach().cpu().numpy()

    def forward(self, current_layer):
        return self.actor.forward(current_layer.float())

    def select_action(self, state):
        state = torch.tensor(state).float()
        return self.tensor_to_array(self.actor.forward(state))

    def model_update(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).float()
        rewards = torch.tensor(rewards).float()
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).float()

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(
                DDPG_FF.get_critic_state(next_states, next_actions)
            )
            y = rewards + self.gamma * target_q * (1 - dones)

        current_q = self.critic(DDPG_FF.get_critic_state(states, actions))
        critic_loss = F.mse_loss(current_q, y)
        DDPG_FF.backprop(self.critic_optimizer, critic_loss)

        actor_loss = -self.critic(
            DDPG_FF.get_critic_state(states, self.actor(states))
        ).mean()
        DDPG_FF.backprop(self.actor_optimizer, actor_loss)

        self.soft_update_targets()

        return critic_loss.item(), actor_loss.item()

    def train(self, states, actions, rewards, next_states, dones, epochs):
        total_critic_loss = 0
        total_actor_loss = 0

        for i in range(epochs):
            critic_loss, actor_loss = self.model_update(
                states, actions, rewards, next_states, dones
            )
            total_critic_loss += critic_loss
            total_actor_loss += actor_loss

        avg_critic_loss = total_critic_loss / epochs
        avg_actor_loss = total_actor_loss / epochs
        return avg_critic_loss, avg_actor_loss