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

    def compute_rolling_correlation(self, series_a: pd.Series, series_b: pd.Series, window: int = 60) -> pd.Series:
        """Compute rolling correlation between two price series.

        Args:
            series_a: First price series (returns computed internally)
            series_b: Second price series
            window: Rolling window

        Returns:
            Series of rolling correlations
        """
        ret_a = series_a.pct_change()
        ret_b = series_b.pct_change()
        return ret_a.rolling(window=window).corr(ret_b)

    def compute_all(self, df: pd.DataFrame, vix_data: pd.Series = None) -> pd.DataFrame:
        """Compute all indicators and return enriched DataFrame.

        Args:
            df: OHLCV DataFrame
            vix_data: Optional VIX close price Series (indexed by date)

        Returns:
            DataFrame with all indicator columns added
        """
        result = df.copy()
        close = result["Close"]
        returns = close.pct_change()

        result["zscore"] = self.compute_zscore(close)
        result = self.compute_bollinger_bands(result)
        result["rsi"] = self.compute_rsi(close)
        macd_df = self.compute_macd(close)
        result["macd"] = macd_df["macd"]
        result["macd_signal"] = macd_df["macd_signal"]
        result["macd_hist"] = macd_df["macd_hist"]
        result["atr"] = self.compute_atr(result)
        result["volume_zscore"] = self.compute_volume_zscore(result["Volume"])
        result["volatility"] = self.compute_rolling_volatility(close)
        result["roc"] = self.compute_roc(close)

        for lag in [1, 2, 3, 5, 10]:
            result[f"return_lag_{lag}"] = returns.shift(lag)

        result["skewness_20"] = returns.rolling(window=20).skew()
        result["kurtosis_20"] = returns.rolling(window=20).kurt()

        vol_5 = returns.rolling(window=5).std()
        vol_20 = returns.rolling(window=20).std()
        result["vol_ratio"] = vol_5 / vol_20.replace(0, np.nan)

        vol_ma5 = result["Volume"].rolling(window=5).mean()
        vol_ma20 = result["Volume"].rolling(window=20).mean()
        result["volume_trend"] = vol_ma5 / vol_ma20.replace(0, np.nan)

        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=200).mean()
        result["dist_sma50"] = (close - sma_50) / sma_50.replace(0, np.nan)
        result["dist_sma200"] = (close - sma_200) / sma_200.replace(0, np.nan)

        result["autocorr_10"] = returns.rolling(window=20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
        )

        result["intraday_range"] = (result["High"] - result["Low"]) / close.replace(0, np.nan)
        result["gap"] = (result["Open"] - close.shift(1)) / close.shift(1).replace(0, np.nan)

        result["day_of_week"] = result.index.dayofweek
        result["month"] = result.index.month

        # Rolling return momentum (5d, 10d, 20d) for context
        result["ret_5d"] = close.pct_change(5)
        result["ret_10d"] = close.pct_change(10)
        result["ret_20d"] = close.pct_change(20)

        # Volatility percentile (for regime context)
        result["vol_percentile"] = vol_20.rolling(window=252, min_periods=20).rank(pct=True)

        # VIX integration (macro filter)
        if vix_data is not None:
            vix_aligned = vix_data.reindex(result.index).ffill()
            result["vix"] = vix_aligned
            result["vix_sma20"] = vix_aligned.rolling(window=20).mean()
            result["vix_zscore"] = self.compute_zscore(vix_aligned, window=20)

        return result
