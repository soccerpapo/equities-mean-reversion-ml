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
