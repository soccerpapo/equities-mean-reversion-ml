import numpy as np
import pandas as pd
import pytest
from strategy.regime_detector import RegimeDetector


def make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame large enough for regime detection."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.abs(close) + 1  # ensure positive prices
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def detector():
    return RegimeDetector(n_components=3, random_state=0)


@pytest.fixture
def fitted_detector():
    df = make_df(300)
    d = RegimeDetector(n_components=3, random_state=0)
    d.fit(df)
    return d, df


class TestRegimeDetectorFit:
    def test_fit_returns_self(self, detector):
        df = make_df(300)
        result = detector.fit(df)
        assert result is detector

    def test_is_fitted_after_fit(self, detector):
        df = make_df(300)
        detector.fit(df)
        assert detector._is_fitted is True

    def test_regime_order_has_correct_length(self, fitted_detector):
        detector, _ = fitted_detector
        assert len(detector._sorted_gmm_labels) == 3

    def test_regime_order_contains_valid_indices(self, fitted_detector):
        detector, _ = fitted_detector
        assert sorted(detector._sorted_gmm_labels) == [0, 1, 2]


class TestDetectRegime:
    def test_returns_valid_regime_label(self, fitted_detector):
        detector, df = fitted_detector
        regime, confidence = detector.detect_regime(df)
        assert regime in (0, 1, 2)

    def test_confidence_is_between_0_and_1(self, fitted_detector):
        detector, df = fitted_detector
        _, confidence = detector.detect_regime(df)
        assert 0.0 <= confidence <= 1.0

    def test_unfitted_returns_default_regime(self, detector):
        df = make_df(300)
        regime, confidence = detector.detect_regime(df)
        assert regime == 1
        assert confidence == 0.5

    def test_insufficient_data_returns_default(self, fitted_detector):
        detector, _ = fitted_detector
        tiny_df = make_df(5)
        regime, confidence = detector.detect_regime(tiny_df)
        assert regime == 1
        assert confidence == 0.5


class TestGetPositionMultiplier:
    def test_regime_0_returns_full_size(self, detector):
        assert detector.get_position_multiplier(0) == 1.0

    def test_regime_1_returns_half_size(self, detector):
        assert detector.get_position_multiplier(1) == 0.5

    def test_regime_2_returns_zero(self, detector):
        assert detector.get_position_multiplier(2) == 0.0

    def test_unknown_regime_returns_default(self, detector):
        assert detector.get_position_multiplier(99) == 0.5


class TestSaveLoadModel:
    def test_save_and_load_preserves_regime(self, fitted_detector, tmp_path):
        detector, df = fitted_detector
        path = str(tmp_path / "regime_model.joblib")
        detector.save_model(path)

        loaded = RegimeDetector.load_model(path)
        assert loaded._is_fitted is True

        regime_orig, _ = detector.detect_regime(df)
        regime_loaded, _ = loaded.detect_regime(df)
        assert regime_orig == regime_loaded

    def test_save_unfitted_does_not_raise(self, detector, tmp_path):
        path = str(tmp_path / "regime_model.joblib")
        # Should just log a warning and not crash
        detector.save_model(path)
        import os
        assert not os.path.exists(path)
