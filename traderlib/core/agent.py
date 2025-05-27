import torch
import torch.optim as optim
import numpy as np
from .model_arch import PolicyNetwork, ValueNetwork

class RolloutBuffer:
    """
    Buffer to store rollouts for PPO updates.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.__init__()

class PPOAgent:
    """
    A Proximal Policy Optimization agent.
    """
    def __init__(self, state_dim, action_dim, policy_hidden, value_hidden,
                 lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = PolicyNetwork(state_dim, policy_hidden, action_dim)
        self.value_function = ValueNetwork(state_dim, value_hidden)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_function.parameters()),
            lr=lr
        )
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(dist.log_prob(action))
        return action.item()

    def compute_returns(self, next_value):
        returns = []
        discounted = next_value
        for reward, is_term in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_term:
                discounted = 0
            discounted = reward + (self.gamma * discounted)
            returns.insert(0, discounted)
        return returns

    def update(self):
        # Convert lists to tensors
        states = torch.stack(self.buffer.states)
        actions = torch.stack(self.buffer.actions)
        old_log_probs = torch.stack(self.buffer.log_probs)
        returns = torch.tensor(self.compute_returns(0), dtype=torch.float32)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluations
            new_probs = self.policy(states)
            dist = torch.distributions.Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs.detach())

            # Advantages
            values = self.value_function(states)
            advantages = returns - values.detach()

            # Surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() +                    0.5 * (returns - values).pow(2).mean() -                    0.01 * entropy

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear buffer
        self.buffer.clear()
