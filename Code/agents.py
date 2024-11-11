
import numpy as np
import pandas as pd


class BaselineAgent:
    """Base class for trading strategies"""
    def __init__(self):
        self.position = 0
        self.prices = []
        self.volumes = []
        self.positions = []

class BuyAndHoldAgent(BaselineAgent):
    """Simple buy and hold strategy"""
    def select_action(self, state, epsilon=None):
        return 2  # Always long position

class MovingAverageCrossoverAgent(BaselineAgent):
    """Moving Average Crossover strategy"""
    def __init__(self, short_window=50, long_window=200):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window

    def select_action(self, state, epsilon=None):
        self.prices.append(state[0])  # Assuming first state feature is price
        if len(self.prices) < self.long_window:
            return 1  # Neutral until enough data

        short_ma = np.mean(self.prices[-self.short_window:])
        long_ma = np.mean(self.prices[-self.long_window:])

        if short_ma > long_ma:
            return 2  # Long
        elif short_ma < long_ma:
            return 0  # Short
        return 1  # Neutral

class RSIAgent(BaselineAgent):
    """Relative Strength Index strategy"""
    def __init__(self, period=14, overbought=70, oversold=30):
        super().__init__()
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def calculate_rsi(self):
        if len(self.prices) < self.period + 1:
            return 50

        deltas = np.diff(self.prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def select_action(self, state, epsilon=None):
        self.prices.append(state[0])
        rsi = self.calculate_rsi()

        if rsi > self.overbought:
            return 0  # Short when overbought
        elif rsi < self.oversold:
            return 2  # Long when oversold
        return 1  # Neutral otherwise

class BollingerBandsAgent(BaselineAgent):
    """Bollinger Bands strategy"""
    def __init__(self, window=20, num_std=2):
        super().__init__()
        self.window = window
        self.num_std = num_std

    def select_action(self, state, epsilon=None):
        self.prices.append(state[0])
        if len(self.prices) < self.window:
            return 1

        rolling_mean = np.mean(self.prices[-self.window:])
        rolling_std = np.std(self.prices[-self.window:])

        upper_band = rolling_mean + (self.num_std * rolling_std)
        lower_band = rolling_mean - (self.num_std * rolling_std)
        current_price = self.prices[-1]

        if current_price > upper_band:
            return 0  # Short when above upper band
        elif current_price < lower_band:
            return 2  # Long when below lower band
        return 1  # Neutral when between bands

class MACDAgent(BaselineAgent):
    """Moving Average Convergence Divergence strategy"""
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate_macd(self):
        if len(self.prices) < self.slow_period:
            return 0, 0

        fast_ema = pd.Series(self.prices).ewm(span=self.fast_period).mean().iloc[-1]
        slow_ema = pd.Series(self.prices).ewm(span=self.slow_period).mean().iloc[-1]
        macd = fast_ema - slow_ema

        if len(self.prices) < self.slow_period + self.signal_period:
            return macd, macd

        macd_series = pd.Series(self.prices).ewm(span=self.fast_period).mean() - \
                     pd.Series(self.prices).ewm(span=self.slow_period).mean()
        signal = macd_series.ewm(span=self.signal_period).mean().iloc[-1]

        return macd, signal

    def select_action(self, state, epsilon=None):
        self.prices.append(state[0])
        macd, signal = self.calculate_macd()

        if macd > signal:
            return 2  # Long when MACD above signal
        elif macd < signal:
            return 0  # Short when MACD below signal
        return 1  # Neutral when MACD crosses signal

class VolumeWeightedMAAgent(BaselineAgent):
    """Volume Weighted Moving Average strategy"""
    def __init__(self, window=20):
        super().__init__()
        self.window = window

    def select_action(self, state, epsilon=None):
        self.prices.append(state[0])
        self.volumes.append(state[-1])  # Assuming last state feature is volume

        if len(self.prices) < self.window:
            return 1

        price_array = np.array(self.prices[-self.window:])
        volume_array = np.array(self.volumes[-self.window:])
        vwma = np.sum(price_array * volume_array) / np.sum(volume_array)
        current_price = self.prices[-1]

        if current_price > vwma * 1.02:  # 2% above VWMA
            return 0  # Short
        elif current_price < vwma * 0.98:  # 2% below VWMA
            return 2  # Long
        return 1  # Neutral

class MeanReversionAgent(BaselineAgent):
    """Mean Reversion strategy"""
    def __init__(self, window=20, std_dev=2):
        super().__init__()
        self.window = window
        self.std_dev = std_dev

    def select_action(self, state, epsilon=None):
        self.prices.append(state[0])
        if len(self.prices) < self.window:
            return 1

        mean = np.mean(self.prices[-self.window:])
        std = np.std(self.prices[-self.window:])
        current_price = self.prices[-1]

        z_score = (current_price - mean) / std

        if z_score > self.std_dev:
            return 0  # Short when price too high
        elif z_score < -self.std_dev:
            return 2  # Long when price too low
        return 1  # Neutral otherwise
