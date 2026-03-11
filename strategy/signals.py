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

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def check_volatility_regime(self, df: pd.DataFrame) -> bool:
        """Check if current volatility is in the acceptable trading range.

        Computes 20-day realized volatility and its percentile rank over the
        last 252 trading days.  Returns True when the percentile falls between
        VOL_PERCENTILE_LOW and VOL_PERCENTILE_HIGH (default 20th–80th).

        Args:
            df: DataFrame containing at least a 'Close' column.

        Returns:
            True if volatility is in the acceptable range for mean reversion.
        """
        close = df["Close"]
        returns = close.pct_change()
        vol_20 = returns.rolling(window=20).std()

        current_vol = vol_20.iloc[-1]
        if pd.isna(current_vol):
            return True  # insufficient data — allow trading

        lookback = vol_20.dropna().iloc[-252:]
        if len(lookback) < 20:
            return True  # not enough history — allow trading

        percentile = (lookback < current_vol).mean() * 100
        low = getattr(settings, "VOL_PERCENTILE_LOW", 20)
        high = getattr(settings, "VOL_PERCENTILE_HIGH", 80)
        return bool(low <= percentile <= high)

    def check_trend_filter(self, df: pd.DataFrame, side: str) -> bool:
        """Check if a trade aligns with the major trend.

        Computes the 200-day SMA of Close.  For a 'long' (BUY) trade the
        current close must be *above* the SMA; for a 'short' (SELL) trade it
        must be *below* the SMA.

        Args:
            df: DataFrame with 'Close' column.
            side: 'long' or 'short'.

        Returns:
            True if the trade aligns with the trend.
        """
        period = getattr(settings, "TREND_SMA_PERIOD", 200)
        close = df["Close"]
        if len(close) < period:
            return True  # not enough data — allow trading

        sma = close.rolling(window=period).mean().iloc[-1]
        current = close.iloc[-1]
        if pd.isna(sma):
            return True

        if side == "long":
            return bool(current > sma)
        elif side == "short":
            return bool(current < sma)
        return True

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate BUY/SELL/HOLD signals based on multiple indicator confirmation.

        Signal logic:
        - BUY: z-score crosses below -threshold PLUS at least 1 of 3 other confirmations
          (RSI oversold, price near lower BB, high volume). Signal strength 0-1.
        - SELL: z-score crosses above +threshold PLUS at least 1 of 3 other confirmations.
        - Momentum divergence BUY: price making new low while RSI makes higher low.

        When USE_TREND_FILTER is True signals that go against the 200-day SMA are
        suppressed.  Only signals with signal_strength >= MIN_SIGNAL_STRENGTH are kept.

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

        # Weights for each indicator confirmation (sum = 1.0)
        WEIGHTS = {
            "zscore": 0.30,
            "rsi": 0.25,
            "bb": 0.25,
            "volume": 0.20,
        }

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

        # Weighted signal strength
        available_weights = {k: WEIGHTS[k] for k in buy_conditions.columns if k in WEIGHTS}
        total_weight = sum(available_weights.values()) or 1.0

        buy_score = sum(
            buy_conditions[k].astype(float) * w for k, w in available_weights.items()
        ) / total_weight
        sell_score = sum(
            sell_conditions[k].astype(float) * w for k, w in available_weights.items()
        ) / total_weight

        # Require z-score (mandatory) + at least 2 of the optional confirmations
        min_optional = 2
        optional_cols = [c for c in buy_conditions.columns if c != "zscore"]
        has_optional = len(optional_cols) > 0
        if has_optional:
            buy_optional_met = buy_conditions[optional_cols].sum(axis=1) >= min_optional
            sell_optional_met = sell_conditions[optional_cols].sum(axis=1) >= min_optional
        else:
            buy_optional_met = pd.Series(True, index=result.index)
            sell_optional_met = pd.Series(True, index=result.index)

        buy_signal = buy_conditions["zscore"] & buy_optional_met
        sell_signal = sell_conditions["zscore"] & sell_optional_met

        signals[buy_signal] = 1
        signals[sell_signal] = -1
        signal_strength[buy_signal] = buy_score[buy_signal]
        signal_strength[sell_signal] = sell_score[sell_signal]

        # Momentum divergence: price making new 10-day low but RSI is higher than
        # it was when price last made a 10-day low (bullish divergence proxy).
        if "rsi" in result.columns:
            close = result["Close"]
            rsi = result["rsi"]
            price_new_low = close == close.rolling(window=10).min()
            rsi_higher_than_lagged = rsi > rsi.shift(5)
            divergence_buy = price_new_low & rsi_higher_than_lagged & (signals == 0)
            signals[divergence_buy] = 1
            signal_strength[divergence_buy] = 0.4

        # Apply trend filter (row-wise) — suppress signals that fight the 200-day SMA
        use_trend = getattr(settings, "USE_TREND_FILTER", False)
        if use_trend and "Close" in result.columns:
            period = getattr(settings, "TREND_SMA_PERIOD", 200)
            sma = result["Close"].rolling(window=period).mean()
            # BUY only when price is above the SMA; SELL only when below
            long_against_trend = (signals == 1) & (result["Close"] < sma)
            short_against_trend = (signals == -1) & (result["Close"] > sma)
            against_trend = long_against_trend | short_against_trend
            signals[against_trend] = 0
            signal_strength[against_trend] = 0.0

        # Apply minimum signal strength filter
        min_strength = getattr(settings, "MIN_SIGNAL_STRENGTH", 0.0)
        if min_strength > 0:
            weak = (signals != 0) & (signal_strength < min_strength)
            signals[weak] = 0
            signal_strength[weak] = 0.0

        result["signal"] = signals
        result["signal_strength"] = signal_strength

        buys = (signals == 1).sum()
        sells = (signals == -1).sum()
        logger.info(f"Generated {buys} BUY and {sells} SELL signals")
        return result
