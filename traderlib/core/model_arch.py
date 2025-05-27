import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """
    A customizable policy network for PPO.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ValueNetwork(nn.Module):
    """
    A customizable value network for PPO.
    """
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)
