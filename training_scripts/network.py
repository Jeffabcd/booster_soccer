import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        n_features,
        n_actions,
        neurons,
        activation_function,
        output_activation=None,
    ):
        super().__init__()
        self.n_features = n_features
        self.neurons = neurons
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.n_actions = n_actions

        self.n_layers = len(self.neurons) + 1
        self.layers = torch.nn.ModuleList()
        for index in range(self.n_layers):
            if index == 0:
                in_dim = n_features
                out_dim = neurons[index]
            elif index == self.n_layers - 1:
                in_dim = neurons[index - 1]
                out_dim = self.n_actions
            else:
                in_dim = neurons[index - 1]
                out_dim = neurons[index]
            self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, current_layer):
        for index, layer in enumerate(self.layers):
            if index < self.n_layers - 1:
                current_layer = self.activation_function(layer(current_layer))
            else:
                current_layer = layer(current_layer)
                if self.output_activation is not None:
                    current_layer = self.output_activation(current_layer)
        return current_layer