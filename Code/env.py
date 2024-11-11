import gymnasium as gym
import numpy as np
import pandas as pd


class TradingEnvironmentV6(gym.Env):
    """Trading Environment with enhanced position balancing"""
    def __init__(self, df):
        super(TradingEnvironmentV6, self).__init__()
        self.df = df
        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(3)

        # Enhanced observation space
        self.window_size = 10
        self.feature_size = 6  # position, volatility, momentum, position_duration, rel_volume, trend
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.feature_size + self.window_size,),  # Total size is fixed
            dtype=np.float32
        )

        self.max_steps = len(df) - 1
        self.transaction_cost = 0.001
        self.holding_penalty = 0.0002
        self.position_imbalance_penalty = 0.0005
        self.max_position_duration = 20

        self.position_counts = {-1: 0, 0: 0, 1: 0}
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = 10000
        self.position = 0
        self.position_duration = 0
        self.last_trade_price = None
        self.position_counts = {-1: 0, 0: 0, 1: 0}

        obs = self._get_observation()
        return obs, {}

    def _calculate_indicators(self):
        window = self.df.iloc[self.current_step-self.window_size:self.current_step+1]

        current_price = window['close'].iloc[-1]
        returns = window['close'].pct_change().fillna(0)
        volatility = returns.std()
        momentum = (current_price - window['close'].iloc[0]) / window['close'].iloc[0]

        volume_ma = window['volume'].mean()
        relative_volume = window['volume'].iloc[-1] / volume_ma if volume_ma != 0 else 1

        sma_short = window['close'].rolling(window=5).mean().iloc[-1]
        sma_long = window['close'].rolling(window=10).mean().iloc[-1]
        trend = (sma_short / sma_long) - 1 if sma_long != 0 else 0

        return returns.values[:-1], volatility, momentum, relative_volume, trend  # Remove last return to match window size

    def _get_observation(self):
        returns, volatility, momentum, rel_volume, trend = self._calculate_indicators()

        # Ensure returns length matches window_size
        if len(returns) != self.window_size:
            print(f"Warning: returns length {len(returns)} != window_size {self.window_size}")
            returns = returns[:self.window_size]  # Truncate if necessary

        observation = np.concatenate([
            [self.position],
            [volatility],
            [momentum],
            [self.position_duration / self.max_position_duration],
            [rel_volume],
            [trend],
            returns
        ]).astype(np.float32)

        if observation.shape[0] != self.feature_size + self.window_size:
            print(f"Warning: observation shape {observation.shape} doesn't match expected {self.feature_size + self.window_size}")

        return observation

    # Rest of the class remains the same

    def step(self, action):
        new_position = action - 1

        # Get prices and calculate returns
        current_price = self.df['close'].iloc[self.current_step]
        next_price = self.df['close'].iloc[self.current_step + 1]
        price_change = (next_price - current_price) / current_price

        # Update position tracking
        if new_position == self.position:
            self.position_duration += 1
        else:
            self.position_duration = 0
            if self.last_trade_price is not None:
                self.last_trade_price = current_price

        # Update position counts
        self.position_counts[new_position] += 1

        # Calculate position imbalance penalty
        total_positions = sum(self.position_counts.values())
        max_position_count = max(self.position_counts.values())
        position_imbalance = (max_position_count / total_positions) if total_positions > 0 else 0
        imbalance_penalty = self.position_imbalance_penalty * position_imbalance

        # Calculate rewards
        position_reward = self.position * price_change
        transaction_cost = abs(new_position - self.position) * self.transaction_cost
        holding_penalty = self.holding_penalty * (self.position_duration / self.max_position_duration)

        # Direction reward
        direction_reward = 0.002 if (new_position * price_change > 0) else -0.001

        # Diversity reward (encourage changing positions when profitable)
        diversity_reward = 0.001 if (new_position != self.position and
                                   abs(price_change) > self.transaction_cost) else 0

        # Risk management reward
        risk_reward = -0.002 if self.position_duration > self.max_position_duration else 0

        # Total reward
        reward = (position_reward +
                 direction_reward +
                 diversity_reward +
                 risk_reward -
                 transaction_cost -
                 holding_penalty -
                 imbalance_penalty)

        # Update state
        self.position = new_position
        self.portfolio_value *= (1 + reward)
        self.current_step += 1

        # Check if done
        done = self.current_step >= self.max_steps

        return self._get_observation(), reward, done, False, {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'transaction_cost': transaction_cost,
            'position_duration': self.position_duration,
            'position_counts': self.position_counts
        }

