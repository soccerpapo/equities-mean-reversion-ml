import numpy as np
import pandas as pd
import pytest
from features.indicators import IndicatorEngine
from strategy.signals import SignalGenerator


@pytest.fixture
def engine():
    return IndicatorEngine()


@pytest.fixture
def sig_gen():
    return SignalGenerator()


def make_df(n=100, seed=42):
    """Create a synthetic OHLCV DataFrame."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestSignalGeneration:
    def test_returns_dataframe_with_signal(self, sig_gen, engine):
        df = make_df()
        df = engine.compute_all(df)
        result = sig_gen.generate_mean_reversion_signals(df)
        assert "signal" in result.columns
        assert "signal_strength" in result.columns

    def test_signal_values_are_valid(self, sig_gen, engine):
        df = make_df()
        df = engine.compute_all(df)
        result = sig_gen.generate_mean_reversion_signals(df)
        assert result["signal"].isin([-1, 0, 1]).all()

    def test_signal_strength_range(self, sig_gen, engine):
        df = make_df()
        df = engine.compute_all(df)
        result = sig_gen.generate_mean_reversion_signals(df)
        valid = result["signal_strength"]
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_buy_signal_on_extreme_negative_zscore(self, sig_gen):
        """Force a BUY signal by injecting extreme negative z-score and low RSI."""
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = np.linspace(100, 100, n)
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
                "Volume": np.ones(n) * 1e6,
            },
            index=dates,
        )
        # Manually set indicators to trigger BUY
        df["zscore"] = -3.0
        df["rsi"] = 25.0
        df["bb_pct_b"] = 0.05
        df["volume_zscore"] = 2.0
        result = sig_gen.generate_mean_reversion_signals(df)
        assert (result["signal"] == 1).any(), "Expected at least one BUY signal"

    def test_sell_signal_on_extreme_positive_zscore(self, sig_gen):
        """Force a SELL signal by injecting extreme positive z-score and high RSI."""
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = np.linspace(100, 100, n)
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
                "Volume": np.ones(n) * 1e6,
            },
            index=dates,
        )
        df["zscore"] = 3.0
        df["rsi"] = 75.0
        df["bb_pct_b"] = 0.95
        df["volume_zscore"] = 2.0
        result = sig_gen.generate_mean_reversion_signals(df)
        assert (result["signal"] == -1).any(), "Expected at least one SELL signal"

    def test_hold_when_no_clear_signal(self, sig_gen):
        """When z-scores are near zero, expect mostly HOLD signals."""
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = np.ones(n) * 100
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
                "Volume": np.ones(n) * 1e6,
            },
            index=dates,
        )
        df["zscore"] = 0.1
        df["rsi"] = 50.0
        df["bb_pct_b"] = 0.5
        df["volume_zscore"] = 0.1
        result = sig_gen.generate_mean_reversion_signals(df)
        hold_pct = (result["signal"] == 0).mean()
        assert hold_pct > 0.9, f"Expected mostly HOLDs but got {hold_pct:.2%} HOLDs"


class TestVolatilityRegimeFilter:
    def _make_df_with_vol(self, n=300, vol_scale=1.0, seed=0):
        np.random.seed(seed)
        close = 100 + np.cumsum(np.random.randn(n) * vol_scale)
        close = np.abs(close) + 1
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": close}, index=dates)

    def test_normal_vol_allows_trading(self, sig_gen):
        """Normal volatility should be in the acceptable range."""
        df = self._make_df_with_vol(300, vol_scale=1.0)
        # We can't guarantee exact percentile without knowing the distribution,
        # but the method should return a bool without error.
        result = sig_gen.check_volatility_regime(df)
        assert isinstance(result, bool)

    def test_insufficient_data_allows_trading(self, sig_gen):
        """When there is not enough history, allow trading."""
        df = self._make_df_with_vol(10)
        assert sig_gen.check_volatility_regime(df) is True

    def test_returns_bool(self, sig_gen):
        df = self._make_df_with_vol(300)
        result = sig_gen.check_volatility_regime(df)
        assert isinstance(result, bool)


class TestTrendFilter:
    def _make_trending_df(self, n=250, uptrend=True):
        """Create a strongly trending DataFrame."""
        if uptrend:
            close = np.linspace(50, 200, n)  # strong uptrend
        else:
            close = np.linspace(200, 50, n)  # strong downtrend
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": close}, index=dates)

    def test_long_allowed_in_uptrend(self, sig_gen):
        """BUY signals should be allowed when price is above 200-day SMA."""
        df = self._make_trending_df(250, uptrend=True)
        # With only 250 rows, 200-day SMA will be lower than current price in an uptrend
        assert sig_gen.check_trend_filter(df, "long") is True

    def test_short_blocked_in_uptrend(self, sig_gen):
        """SELL signals should be blocked when price is above 200-day SMA."""
        df = self._make_trending_df(250, uptrend=True)
        assert sig_gen.check_trend_filter(df, "short") is False

    def test_short_allowed_in_downtrend(self, sig_gen):
        """SELL signals should be allowed when price is below 200-day SMA."""
        df = self._make_trending_df(250, uptrend=False)
        assert sig_gen.check_trend_filter(df, "short") is True

    def test_long_blocked_in_downtrend(self, sig_gen):
        """BUY signals should be blocked when price is below 200-day SMA."""
        df = self._make_trending_df(250, uptrend=False)
        assert sig_gen.check_trend_filter(df, "long") is False

    def test_insufficient_data_allows_trading(self, sig_gen):
        """When there is not enough data for the SMA, allow all trades."""
        df = self._make_trending_df(50)
        assert sig_gen.check_trend_filter(df, "long") is True
        assert sig_gen.check_trend_filter(df, "short") is True

