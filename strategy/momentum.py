"""Momentum / Trend-Following strategy.

This module implements a momentum-based trading strategy that:
  1. Detects trends using dual moving average crossovers, breakouts, and ADX
  2. Scores stocks by composite momentum (1M, 3M, 6M, 12M returns)
  3. Ranks stocks and selects top-N for long positions
  4. Manages positions with ATR-based trailing stops
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class MomentumTrader:
    """Momentum / trend-following strategy.

    Signals:
      BUY  when: momentum_score > 0.4 AND price > 200 SMA AND ADX > 25
      SELL when: momentum_score < -0.3 OR price < 200 SMA OR trailing stop hit
    """

    def __init__(
        self,
        sma_fast: int = None,
        sma_slow: int = None,
        adx_threshold: float = None,
        top_n: int = None,
        trailing_stop_atr_mult: float = None,
    ):
        """Initialise MomentumTrader with configurable parameters.

        Args:
            sma_fast: Fast SMA period (default from settings).
            sma_slow: Slow SMA period (default from settings).
            adx_threshold: Minimum ADX for trend strength filter (default from settings).
            top_n: Number of top stocks to hold (default from settings).
            trailing_stop_atr_mult: ATR multiplier for trailing stop (default from settings).
        """
        self.sma_fast = sma_fast if sma_fast is not None else getattr(settings, "MOMENTUM_SMA_FAST", 50)
        self.sma_slow = sma_slow if sma_slow is not None else getattr(settings, "MOMENTUM_SMA_SLOW", 200)
        self.adx_threshold = adx_threshold if adx_threshold is not None else getattr(settings, "MOMENTUM_ADX_THRESHOLD", 25)
        self.top_n = top_n if top_n is not None else getattr(settings, "MOMENTUM_TOP_N", 3)
        self.trailing_stop_atr_mult = (
            trailing_stop_atr_mult if trailing_stop_atr_mult is not None
            else getattr(settings, "TRAILING_STOP_ATR_MULT", 2.0)
        )

    # ------------------------------------------------------------------
    # Trend Detection
    # ------------------------------------------------------------------

    def calculate_trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based signals using multiple confirming indicators.

        Combines:
          - Dual MA crossover (50/200-day SMA)
          - 52-week breakout
          - ADX trend strength filter

        Args:
            df: OHLCV DataFrame with at least 'Close', 'High', 'Low' columns.

        Returns:
            DataFrame with additional columns:
              sma_fast, sma_slow, ma_signal,
              high_52w, low_52w, breakout_signal,
              adx, adx_signal,
              trend_score (sum of active confirmations, -1 to +1 scale).
        """
        result = df.copy()

        # --- Dual Moving Average ---
        result["sma_fast"] = result["Close"].rolling(window=self.sma_fast).mean()
        result["sma_slow"] = result["Close"].rolling(window=self.sma_slow).mean()

        # Golden cross (fast > slow) → +1, Death cross (fast < slow) → -1
        result["ma_signal"] = 0
        result.loc[result["sma_fast"] > result["sma_slow"], "ma_signal"] = 1
        result.loc[result["sma_fast"] < result["sma_slow"], "ma_signal"] = -1

        # --- 52-Week Breakout ---
        rolling_52w = 252
        result["high_52w"] = result["Close"].rolling(window=rolling_52w).max().shift(1)
        result["low_52w"] = result["Close"].rolling(window=rolling_52w).min().shift(1)

        result["breakout_signal"] = 0
        result.loc[result["Close"] > result["high_52w"], "breakout_signal"] = 1
        result.loc[result["Close"] < result["low_52w"], "breakout_signal"] = -1

        # --- ADX (Average Directional Index) ---
        result["adx"] = self._calculate_adx(result)
        result["adx_signal"] = (result["adx"] > self.adx_threshold).astype(int)

        # --- Trend Score: combine signals ---
        # Each confirmation adds strength; ADX only gates (no direction)
        result["trend_score"] = (result["ma_signal"] + result["breakout_signal"]) / 2.0
        # Zero out when trend is not strong enough
        result.loc[result["adx"] <= self.adx_threshold, "trend_score"] = 0.0

        return result

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate the Average Directional Index (ADX).

        Args:
            df: DataFrame with High, Low, Close columns.
            period: ADX smoothing period (default 14).

        Returns:
            Series of ADX values.
        """
        if "High" not in df.columns or "Low" not in df.columns:
            return pd.Series(0.0, index=df.index)

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()

        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        pos_dm_series = pd.Series(pos_dm, index=df.index)
        neg_dm_series = pd.Series(neg_dm, index=df.index)

        # Smoothed averages
        atr_smooth = tr.ewm(span=period, adjust=False).mean()
        pos_di = 100 * pos_dm_series.ewm(span=period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
        neg_di = 100 * neg_dm_series.ewm(span=period, adjust=False).mean() / atr_smooth.replace(0, np.nan)

        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx.fillna(0.0)

    # ------------------------------------------------------------------
    # Momentum Scoring
    # ------------------------------------------------------------------

    def calculate_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite momentum score from multi-period returns.

        Weights:
          1-month return:  0.3
          3-month return:  0.3
          6-month return:  0.2
          12-month return: 0.2

        The raw weighted return is normalised to [-1, +1] using a tanh
        transformation so that outliers are dampened.

        Args:
            df: DataFrame with a 'Close' column.

        Returns:
            Series of momentum scores in [-1, +1], same index as df.
        """
        close = df["Close"]

        ret_1m = close.pct_change(21)    # ~1 month
        ret_3m = close.pct_change(63)    # ~3 months
        ret_6m = close.pct_change(126)   # ~6 months
        ret_12m = close.pct_change(252)  # ~12 months

        composite = (
            0.3 * ret_1m.fillna(0)
            + 0.3 * ret_3m.fillna(0)
            + 0.2 * ret_6m.fillna(0)
            + 0.2 * ret_12m.fillna(0)
        )

        # Normalise to [-1, +1] using tanh
        score = np.tanh(composite * 5)
        return pd.Series(score, index=df.index, name="momentum_score")

    # ------------------------------------------------------------------
    # Position Management
    # ------------------------------------------------------------------

    def rank_and_select(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        top_n: int = None,
    ) -> Tuple[List[str], List[str]]:
        """Rank symbols by momentum score and return top / bottom selections.

        Args:
            symbols_data: Dict mapping symbol → OHLCV DataFrame.
            top_n: Number of top stocks to select (defaults to self.top_n).

        Returns:
            Tuple (top_symbols, bottom_symbols) for long and short candidates.
        """
        top_n = top_n or self.top_n
        scores = {}
        for symbol, df in symbols_data.items():
            if df.empty or "Close" not in df.columns:
                continue
            score_series = self.calculate_momentum_score(df)
            if not score_series.empty and not pd.isna(score_series.iloc[-1]):
                scores[symbol] = float(score_series.iloc[-1])

        if not scores:
            return [], []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, _ in ranked[:top_n]]
        bottom_symbols = [sym for sym, _ in ranked[-top_n:]]
        logger.info(
            "Momentum ranking: top=%s, bottom=%s", top_symbols, bottom_symbols
        )
        return top_symbols, bottom_symbols

    def calculate_trailing_stop(
        self,
        entry_price: float,
        highest_since_entry: float,
        atr: float,
        multiplier: float = None,
    ) -> float:
        """Calculate the current trailing stop price.

        For a long position, the stop is placed at:
          max(entry_price - mult * ATR, highest_since_entry - mult * ATR)

        Args:
            entry_price: Price at which the position was entered.
            highest_since_entry: Highest price recorded since entry.
            atr: Current ATR value.
            multiplier: ATR multiplier (defaults to self.trailing_stop_atr_mult).

        Returns:
            Trailing stop price.
        """
        multiplier = multiplier if multiplier is not None else self.trailing_stop_atr_mult
        if atr <= 0:
            return entry_price * 0.95  # fallback: 5% below entry
        stop_from_entry = entry_price - multiplier * atr
        stop_from_high = highest_since_entry - multiplier * atr
        return max(stop_from_entry, stop_from_high)

    # ------------------------------------------------------------------
    # Signal Generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum trading signals for a single symbol using Unsupervised Clustering (GMM).

        BUY  when: momentum_score is in the 'Extreme' cluster AND price > SMA_SLOW AND ADX > threshold
        SELL when: momentum_score drops below the 'Extreme' cluster boundary OR price < SMA_SLOW OR trailing stop hit

        Args:
            df: OHLCV DataFrame with 'Close', 'High', 'Low', optionally 'atr'.

        Returns:
            DataFrame with additional columns:
              signal (+1 buy, -1 sell, 0 hold),
              signal_strength (0-1),
              momentum_score,
              dynamic_threshold,
              trailing_stop,
              trend_score.
        """
        from sklearn.mixture import GaussianMixture
        
        result = self.calculate_trend_signals(df)
        result["momentum_score"] = self.calculate_momentum_score(df)

        # ATR for trailing stop
        if "atr" not in result.columns:
            high_low = result["High"] - result["Low"] if "High" in result.columns and "Low" in result.columns else pd.Series(0, index=result.index)
            result["atr"] = high_low.rolling(window=14).mean().fillna(0)

        # Calculate Dynamic Thresholds via Rolling GMM
        mom_array = result["momentum_score"].fillna(0).values
        dynamic_thresholds = np.zeros(len(mom_array))
        
        for i in range(len(mom_array)):
            if i < 126: # Need at least 6 months of data to cluster
                dynamic_thresholds[i] = 0.4 # fallback
                continue
                
            # Strictly T-1 lookback to prevent look-ahead bias
            start_idx = max(0, i - 252)
            window_data = mom_array[start_idx:i].reshape(-1, 1) 
            
            try:
                # 2-cluster GMM (Weak/Normal vs Extreme Momentum)
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(window_data)
                
                # Find the cluster with the highest mean (Extreme Momentum)
                extreme_cluster_idx = np.argmax(gmm.means_)
                labels = gmm.predict(window_data)
                extreme_scores = window_data[labels == extreme_cluster_idx]
                
                if len(extreme_scores) > 0:
                    threshold = extreme_scores.min()
                    # Sanity check: Ensure threshold isn't completely negative
                    dynamic_thresholds[i] = max(0.1, threshold)
                else:
                    dynamic_thresholds[i] = 0.4
            except Exception:
                dynamic_thresholds[i] = 0.4
                
        result["dynamic_threshold"] = dynamic_thresholds

        # Trailing stop tracking
        entry_price = 0.0
        highest_since_entry = 0.0
        in_position = False
        trailing_stops = pd.Series(np.nan, index=result.index)
        POST_SELL_COOLDOWN_BARS = 5
        cooldown_remaining = 0

        # Initialise signal columns
        result["signal"] = 0
        result["signal_strength"] = 0.0

        for idx, i in zip(result.index, range(len(result))):
            row = result.loc[idx]
            price = float(row["Close"])
            mom_score = float(row["momentum_score"]) if not pd.isna(row["momentum_score"]) else 0.0
            sma_slow_val = float(row["sma_slow"]) if not pd.isna(row.get("sma_slow", np.nan)) else np.nan
            adx_val = float(row["adx"]) if not pd.isna(row.get("adx", np.nan)) else 0.0
            atr_val = float(row["atr"]) if not pd.isna(row.get("atr", np.nan)) else 0.0
            
            current_threshold = dynamic_thresholds[i]

            if in_position:
                if price > highest_since_entry:
                    highest_since_entry = price
                stop_price = self.calculate_trailing_stop(entry_price, highest_since_entry, atr_val)
                trailing_stops[idx] = stop_price

                # Exit conditions
                stop_hit = price <= stop_price
                below_sma = (not pd.isna(sma_slow_val)) and (price < sma_slow_val)
                # Exit if momentum drops below the dynamic "Extreme" cluster boundary
                weak_momentum = mom_score < (current_threshold - 0.1)

                if stop_hit or below_sma or weak_momentum:
                    result.at[idx, "signal"] = -1
                    result.at[idx, "signal_strength"] = abs(mom_score)
                    in_position = False
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    cooldown_remaining = POST_SELL_COOLDOWN_BARS
            else:
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    continue

                # Entry conditions
                above_sma = (not pd.isna(sma_slow_val)) and (price > sma_slow_val)
                strong_trend = adx_val > self.adx_threshold
                strong_momentum = mom_score > current_threshold

                if strong_momentum and above_sma and strong_trend:
                    result.at[idx, "signal"] = 1
                    # Scale strength
                    strength = min(1.0, 0.3 + max(0, mom_score - current_threshold))
                    result.at[idx, "signal_strength"] = round(strength, 4)
                    in_position = True
                    entry_price = price
                    highest_since_entry = price
                    trailing_stops[idx] = self.calculate_trailing_stop(price, price, atr_val)

        result["trailing_stop"] = trailing_stops
        return result
