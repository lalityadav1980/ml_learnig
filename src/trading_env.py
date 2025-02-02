import gym
from gym import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning that uses historical data.
    The state includes features such as ADX, TR, ATR, bands, a binary representation of SuperTrend,
    and reward. The action is a discrete trading decision:
      0: NoTrade
      1: Buy
      2: Sell
    """

    def __init__(self, data: pd.DataFrame):
        super(TradingEnv, self).__init__()
        # Reset index if the date is the index and then keep it as a column if desired.
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.total_steps = len(self.data)

        # Define the observation space.
        # For example, state features: ADX, TR, ATR, Basic UB, Basic LB, Final UB, Final LB,
        # a numeric representation of SuperTrend (1 for up, 0 for down), and reward.
        obs_shape = (9,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # Define the action space: 0: NoTrade, 1: Buy, 2: Sell.
        self.action_space = spaces.Discrete(3)

        # For simulation purposes, track trade state.
        self.trade_open = False
        self.entry_price = None

    def _get_observation(self):
        # Extract the features from the current row. Adjust the feature order as needed.
        row = self.data.iloc[self.current_step]
        obs = np.array([
            row['ADX'],
            row['TR'],
            row['ATR'],
            row['Basic Upper Band'],
            row['Basic Lower Band'],
            row['Final Upper Band'],
            row['Final Lower Band'],
            1.0 if row['STX'] == 'up' else 0.0,
            row['reward']
        ], dtype=np.float32)
        return obs

    def step(self, action):
        """
        Executes one time step.
        Action:
          0: NoTrade,
          1: Buy,
          2: Sell.
        Reward is computed based on the simulated profit if a trade is closed.
        """
        done = False
        current_row = self.data.iloc[self.current_step]
        current_close = current_row['close']
        reward = 0.0

        # Simple trade simulation logic:
        if action == 1:  # Buy
            if not self.trade_open:
                self.trade_open = True
                self.entry_price = current_close
            # If already in a trade, no additional reward change.
        elif action == 2:  # Sell
            if self.trade_open:
                # Assume a Buy trade: exit trade and compute profit.
                reward = current_close - self.entry_price
                self.trade_open = False
                self.entry_price = None
            # Optionally, if not in trade, you might choose to penalize.
        else:  # NoTrade
            if self.trade_open:
                # Optionally, close the trade on a NoTrade decision.
                reward = current_close - self.entry_price
                self.trade_open = False
                self.entry_price = None

        # Advance the time step.
        self.current_step += 1
        if self.current_step >= self.total_steps:
            done = True

        next_state = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return next_state, reward, done, info

    def reset(self):
        self.current_step = 0
        self.trade_open = False
        self.entry_price = None
        return self._get_observation()

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Trade Open: {self.trade_open} | Entry Price: {self.entry_price}")
