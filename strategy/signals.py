import logging
import numpy as np
import pandas as pd
from config import settings
from features.indicators import IndicatorEngine

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates mean reversion trading signals."""

    def __init__(self):
        self._engine = IndicatorEngine()

    def generate_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate BUY/SELL/HOLD signals based on multiple indicator confirmation.

        Signal logic:
        - BUY: z-score crosses below -threshold, RSI oversold, price near lower BB
        - SELL: z-score crosses above +threshold, RSI overbought, price near upper BB
        - Signal strength is a 0-1 float based on number of confirmations

        Args:
            df: DataFrame with computed indicators (or raw OHLCV)

        Returns:
            DataFrame with 'signal' (1=BUY, -1=SELL, 0=HOLD) and 'signal_strength' columns
        """
        result = df.copy()

        if "zscore" not in result.columns:
            result = self._engine.compute_all(result)

        signals = pd.Series(0, index=result.index)
        signal_strength = pd.Series(0.0, index=result.index)

        entry_thresh = settings.Z_SCORE_ENTRY_THRESHOLD
        rsi_oversold = settings.RSI_OVERSOLD
        rsi_overbought = settings.RSI_OVERBOUGHT

        buy_conditions = pd.DataFrame(index=result.index)
        sell_conditions = pd.DataFrame(index=result.index)

        # Z-score conditions (mandatory)
        buy_conditions["zscore"] = result["zscore"] < -entry_thresh
        sell_conditions["zscore"] = result["zscore"] > entry_thresh

        # RSI conditions
        if "rsi" in result.columns:
            buy_conditions["rsi"] = result["rsi"] < rsi_oversold
            sell_conditions["rsi"] = result["rsi"] > rsi_overbought

        # Bollinger Band conditions
        if "bb_pct_b" in result.columns:
            buy_conditions["bb"] = result["bb_pct_b"] < 0.1
            sell_conditions["bb"] = result["bb_pct_b"] > 0.9

        # Volume confirmation
        if "volume_zscore" in result.columns:
            buy_conditions["volume"] = result["volume_zscore"] > 1.0
            sell_conditions["volume"] = result["volume_zscore"] > 1.0

        n_conditions = len(buy_conditions.columns)

        buy_score = buy_conditions.sum(axis=1) / n_conditions
        sell_score = sell_conditions.sum(axis=1) / n_conditions

        # Require z-score as mandatory condition plus at least half confirmed
        buy_signal = buy_conditions["zscore"] & (buy_score >= 0.5)
        sell_signal = sell_conditions["zscore"] & (sell_score >= 0.5)

        signals[buy_signal] = 1
        signals[sell_signal] = -1
        signal_strength[buy_signal] = buy_score[buy_signal]
        signal_strength[sell_signal] = sell_score[sell_signal]

        result["signal"] = signals
        result["signal_strength"] = signal_strength

        buys = (signals == 1).sum()
        sells = (signals == -1).sum()
        logger.info(f"Generated {buys} BUY and {sells} SELL signals")
        return result
