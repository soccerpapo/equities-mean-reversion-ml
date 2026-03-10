import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """Computes technical indicators for mean reversion strategy."""

    def compute_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Compute rolling z-score of a price series.

        Args:
            series: Price series
            window: Rolling window size

        Returns:
            Series of z-scores
        """
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std.replace(0, np.nan)

    def compute_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Compute Bollinger Bands.

        Args:
            df: DataFrame with Close column
            window: Rolling window size
            num_std: Number of standard deviations for bands

        Returns:
            DataFrame with bb_upper, bb_lower, bb_middle, bb_pct_b, bb_bandwidth columns
        """
        close = df["Close"]
        rolling_mean = close.rolling(window=window).mean()
        rolling_std = close.rolling(window=window).std()
        upper = rolling_mean + num_std * rolling_std
        lower = rolling_mean - num_std * rolling_std
        band_range = (upper - lower).replace(0, np.nan)
        bandwidth = band_range / rolling_mean.replace(0, np.nan)
        pct_b = (close - lower) / band_range
        result = df.copy()
        result["bb_upper"] = upper
        result["bb_lower"] = lower
        result["bb_middle"] = rolling_mean
        result["bb_pct_b"] = pct_b
        result["bb_bandwidth"] = bandwidth
        return result

    def compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI using exponential weighted method.

        Args:
            series: Price series
            period: RSI period

        Returns:
            Series of RSI values (0-100)
        """
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # When avg_loss is 0, all moves are gains => RSI = 100
        rsi = rsi.where(avg_loss != 0, 100.0)
        return rsi

    def compute_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Compute MACD line, signal line, and histogram.

        Args:
            series: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with macd, macd_signal, macd_hist columns
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        }, index=series.index)

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range.

        Args:
            df: DataFrame with High, Low, Close columns
            period: ATR period

        Returns:
            Series of ATR values
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()

    def compute_volume_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Compute rolling z-score of volume.

        Args:
            series: Volume series
            window: Rolling window size

        Returns:
            Series of volume z-scores
        """
        return self.compute_zscore(series, window=window)

    def compute_rolling_volatility(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Compute rolling standard deviation of returns.

        Args:
            series: Price series
            window: Rolling window size

        Returns:
            Series of rolling volatility
        """
        returns = series.pct_change()
        return returns.rolling(window=window).std()

    def compute_roc(self, series: pd.Series, period: int = 10) -> pd.Series:
        """Compute Rate of Change.

        Args:
            series: Price series
            period: Look-back period

        Returns:
            Series of ROC values (percentage)
        """
        shifted = series.shift(period)
        return (series - shifted) / shifted.replace(0, np.nan) * 100

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and return enriched DataFrame.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with all indicator columns added
        """
        result = df.copy()
        result["zscore"] = self.compute_zscore(result["Close"])
        result = self.compute_bollinger_bands(result)
        result["rsi"] = self.compute_rsi(result["Close"])
        macd_df = self.compute_macd(result["Close"])
        result["macd"] = macd_df["macd"]
        result["macd_signal"] = macd_df["macd_signal"]
        result["macd_hist"] = macd_df["macd_hist"]
        result["atr"] = self.compute_atr(result)
        result["volume_zscore"] = self.compute_volume_zscore(result["Volume"])
        result["volatility"] = self.compute_rolling_volatility(result["Close"])
        result["roc"] = self.compute_roc(result["Close"])
        return result
