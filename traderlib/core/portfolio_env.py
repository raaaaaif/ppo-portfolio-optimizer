import gym
from gym import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    """
    A simple portfolio environment for trading multiple assets.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, price_array, initial_cash=10000):
        super().__init__()
        self.price_array = price_array
        self.n_assets = price_array.shape[1]
        self.initial_cash = initial_cash

        # Actions: allocation to each asset (including cash)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        # Observations: current prices + current portfolio weights
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.n_assets*2,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.cash = self.initial_cash
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        return self._get_obs()

    def step(self, action):
        # Normalize action to weights
        weights = np.clip(action, 0, 1)
        weights = weights / (np.sum(weights) + 1e-8)

        # Compute portfolio value change
        prev_prices = self.price_array[self.step_idx]
        self.step_idx += 1
        curr_prices = self.price_array[self.step_idx]
        returns = (curr_prices - prev_prices) / prev_prices

        # Update portfolio
        portfolio_return = np.dot(weights, returns)
        self.cash *= (1 + portfolio_return)
        self.weights = weights

        done = self.step_idx >= len(self.price_array) - 1
        reward = self.cash - self.initial_cash
        obs = self._get_obs()
        info = {'cash': self.cash}
        return obs, reward, done, info

    def _get_obs(self):
        prices = self.price_array[self.step_idx]
        return np.concatenate([prices, self.weights], axis=0)

    def render(self, mode='human'):
        print(f"Step: {self.step_idx}, Cash: {self.cash}")
