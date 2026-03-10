import numpy as np
import pandas as pd
import pytest
from features.indicators import IndicatorEngine


@pytest.fixture
def sample_df():
    """Create synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def engine():
    return IndicatorEngine()


class TestZScore:
    def test_zscore_returns_series(self, engine, sample_df):
        result = engine.compute_zscore(sample_df["Close"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)

    def test_zscore_has_nan_at_start(self, engine, sample_df):
        result = engine.compute_zscore(sample_df["Close"], window=20)
        assert result.iloc[:19].isna().all()

    def test_zscore_known_value(self, engine):
        """Test z-score with constant series (should be NaN or 0)."""
        series = pd.Series([10.0] * 30)
        result = engine.compute_zscore(series, window=20)
        # Standard deviation of constant series is 0; result should be NaN
        assert result.iloc[25:].isna().all() or (result.iloc[25:] == 0).all()

    def test_zscore_mean_zero(self, engine):
        """Mean of rolling z-scores should be near zero."""
        series = pd.Series(np.random.randn(200) + 50)
        result = engine.compute_zscore(series, window=20).dropna()
        assert abs(result.mean()) < 0.5


class TestBollingerBands:
    def test_returns_df_with_correct_cols(self, engine, sample_df):
        result = engine.compute_bollinger_bands(sample_df)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_pct_b" in result.columns
        assert "bb_bandwidth" in result.columns

    def test_upper_above_lower(self, engine, sample_df):
        result = engine.compute_bollinger_bands(sample_df)
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_middle_between_bands(self, engine, sample_df):
        result = engine.compute_bollinger_bands(sample_df)
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()


class TestRSI:
    def test_rsi_returns_series(self, engine, sample_df):
        result = engine.compute_rsi(sample_df["Close"])
        assert isinstance(result, pd.Series)

    def test_rsi_range(self, engine, sample_df):
        result = engine.compute_rsi(sample_df["Close"], period=14).dropna()
        assert (result >= 0).all()
        assert (result <= 100).all()

    def test_rsi_oversold_on_downtrend(self, engine):
        series = pd.Series([100 - i * 0.5 for i in range(60)])
        result = engine.compute_rsi(series, period=14).dropna()
        assert result.iloc[-1] < 40

    def test_rsi_overbought_on_uptrend(self, engine):
        series = pd.Series([100 + i * 0.5 for i in range(60)])
        result = engine.compute_rsi(series, period=14).dropna()
        assert result.iloc[-1] > 60


class TestMACD:
    def test_macd_returns_dataframe(self, engine, sample_df):
        result = engine.compute_macd(sample_df["Close"])
        assert isinstance(result, pd.DataFrame)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_macd_hist_is_diff(self, engine, sample_df):
        result = engine.compute_macd(sample_df["Close"])
        diff = result["macd"] - result["macd_signal"]
        pd.testing.assert_series_equal(result["macd_hist"], diff, check_names=False)


class TestATR:
    def test_atr_returns_series(self, engine, sample_df):
        result = engine.compute_atr(sample_df)
        assert isinstance(result, pd.Series)

    def test_atr_positive(self, engine, sample_df):
        result = engine.compute_atr(sample_df, period=14).dropna()
        assert (result > 0).all()


class TestVolumeZScore:
    def test_volume_zscore(self, engine, sample_df):
        result = engine.compute_volume_zscore(sample_df["Volume"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)


class TestRollingVolatility:
    def test_returns_series(self, engine, sample_df):
        result = engine.compute_rolling_volatility(sample_df["Close"])
        assert isinstance(result, pd.Series)

    def test_positive_values(self, engine, sample_df):
        result = engine.compute_rolling_volatility(sample_df["Close"]).dropna()
        assert (result >= 0).all()


class TestROC:
    def test_returns_series(self, engine, sample_df):
        result = engine.compute_roc(sample_df["Close"])
        assert isinstance(result, pd.Series)

    def test_roc_known_value(self, engine):
        series = pd.Series([100.0] * 5 + [110.0] * 5)
        result = engine.compute_roc(series, period=5)
        assert abs(result.iloc[-1] - 10.0) < 0.01


class TestComputeAll:
    def test_compute_all_adds_columns(self, engine, sample_df):
        result = engine.compute_all(sample_df)
        expected_cols = [
            "zscore", "rsi", "macd", "bb_upper", "atr",
            "volume_zscore", "volatility", "roc",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
