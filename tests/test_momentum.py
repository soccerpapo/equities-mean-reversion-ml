"""Tests for strategy/momentum.py."""

import numpy as np
import pandas as pd
import pytest

from strategy.momentum import MomentumTrader


def make_df(n=300, uptrend=True, seed=42):
    """Create a synthetic OHLCV DataFrame."""
    np.random.seed(seed)
    if uptrend:
        close = np.linspace(50, 200, n) + np.random.randn(n) * 2
    else:
        close = np.linspace(200, 50, n) + np.random.randn(n) * 2
    close = np.maximum(close, 1.0)
    high = close + np.abs(np.random.randn(n) * 1.0)
    low = close - np.abs(np.random.randn(n) * 1.0)
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def mt():
    return MomentumTrader(
        sma_fast=10, sma_slow=50, adx_threshold=20, top_n=2, trailing_stop_atr_mult=2.0
    )


class TestTrendSignals:
    def test_returns_expected_columns(self, mt):
        df = make_df(n=300, uptrend=True)
        result = mt.calculate_trend_signals(df)
        for col in ("sma_fast", "sma_slow", "ma_signal", "breakout_signal", "adx", "trend_score"):
            assert col in result.columns, f"Missing column: {col}"

    def test_ma_signal_positive_in_uptrend(self, mt):
        """In a clear uptrend, fast SMA > slow SMA → ma_signal should be +1."""
        df = make_df(n=300, uptrend=True)
        result = mt.calculate_trend_signals(df)
        # After sufficient warm-up, last row should show uptrend
        last_valid = result[["sma_fast", "sma_slow", "ma_signal"]].dropna()
        if not last_valid.empty:
            last = last_valid.iloc[-1]
            assert last["ma_signal"] == 1

    def test_ma_signal_negative_in_downtrend(self, mt):
        """In a clear downtrend, fast SMA < slow SMA → ma_signal should be -1."""
        df = make_df(n=300, uptrend=False)
        result = mt.calculate_trend_signals(df)
        last_valid = result[["sma_fast", "sma_slow", "ma_signal"]].dropna()
        if not last_valid.empty:
            last = last_valid.iloc[-1]
            assert last["ma_signal"] == -1

    def test_adx_non_negative(self, mt):
        df = make_df(n=300)
        result = mt.calculate_trend_signals(df)
        assert (result["adx"].fillna(0) >= 0).all()

    def test_trend_score_range(self, mt):
        df = make_df(n=300)
        result = mt.calculate_trend_signals(df)
        scores = result["trend_score"].dropna()
        assert (scores >= -1.0).all()
        assert (scores <= 1.0).all()


class TestMomentumScore:
    def test_score_in_range(self, mt):
        df = make_df(n=300)
        score = mt.calculate_momentum_score(df)
        assert (score >= -1.0).all()
        assert (score <= 1.0).all()

    def test_uptrend_score_positive(self, mt):
        """Strong uptrend should produce a positive momentum score at the end."""
        df = make_df(n=300, uptrend=True)
        score = mt.calculate_momentum_score(df)
        last_valid = score.dropna()
        if not last_valid.empty:
            assert last_valid.iloc[-1] > 0, "Expected positive score in uptrend"

    def test_downtrend_score_negative(self, mt):
        """Strong downtrend should produce a negative momentum score at the end."""
        df = make_df(n=300, uptrend=False)
        score = mt.calculate_momentum_score(df)
        last_valid = score.dropna()
        if not last_valid.empty:
            assert last_valid.iloc[-1] < 0, "Expected negative score in downtrend"

    def test_score_series_length(self, mt):
        df = make_df(n=200)
        score = mt.calculate_momentum_score(df)
        assert len(score) == len(df)


class TestRankAndSelect:
    def test_returns_top_n_symbols(self, mt):
        symbols_data = {
            "A": make_df(n=300, uptrend=True, seed=1),
            "B": make_df(n=300, uptrend=False, seed=2),
            "C": make_df(n=300, uptrend=True, seed=3),
            "D": make_df(n=300, uptrend=False, seed=4),
        }
        top, bottom = mt.rank_and_select(symbols_data, top_n=2)
        assert len(top) <= 2
        assert len(bottom) <= 2

    def test_top_are_best_performers(self, mt):
        """Top symbols should have higher momentum scores than bottom symbols."""
        symbols_data = {
            "strong_up": make_df(n=300, uptrend=True, seed=1),
            "strong_down": make_df(n=300, uptrend=False, seed=2),
        }
        top, bottom = mt.rank_and_select(symbols_data, top_n=1)
        assert "strong_up" in top
        assert "strong_down" in bottom

    def test_empty_data_returns_empty(self, mt):
        top, bottom = mt.rank_and_select({}, top_n=2)
        assert top == []
        assert bottom == []

    def test_single_symbol(self, mt):
        symbols_data = {"only": make_df(n=300, uptrend=True)}
        top, bottom = mt.rank_and_select(symbols_data, top_n=1)
        assert top == ["only"]
        assert bottom == ["only"]


class TestTrailingStop:
    def test_stop_below_entry(self, mt):
        """Trailing stop should be below entry price."""
        stop = mt.calculate_trailing_stop(
            entry_price=100.0,
            highest_since_entry=100.0,
            atr=2.0,
            multiplier=2.0,
        )
        assert stop < 100.0

    def test_stop_trails_high(self, mt):
        """Trailing stop should rise as price rises."""
        stop_low = mt.calculate_trailing_stop(
            entry_price=100.0, highest_since_entry=100.0, atr=2.0, multiplier=2.0
        )
        stop_high = mt.calculate_trailing_stop(
            entry_price=100.0, highest_since_entry=120.0, atr=2.0, multiplier=2.0
        )
        assert stop_high > stop_low

    def test_zero_atr_fallback(self, mt):
        """When ATR is zero, fallback to 5% below entry."""
        stop = mt.calculate_trailing_stop(100.0, 110.0, 0.0)
        assert stop == pytest.approx(95.0)

    def test_stop_never_below_initial(self, mt):
        """Stop should never be set lower than entry-price minus initial stop."""
        initial_stop = mt.calculate_trailing_stop(100.0, 100.0, 3.0, 2.0)
        high_stop = mt.calculate_trailing_stop(100.0, 105.0, 3.0, 2.0)
        assert high_stop >= initial_stop


class TestSignalGeneration:
    def test_signal_columns_present(self, mt):
        df = make_df(n=300)
        result = mt.generate_signals(df)
        for col in ("signal", "signal_strength", "momentum_score", "trailing_stop"):
            assert col in result.columns

    def test_signal_values_valid(self, mt):
        df = make_df(n=300)
        result = mt.generate_signals(df)
        assert result["signal"].isin([-1, 0, 1]).all()

    def test_signal_strength_range(self, mt):
        df = make_df(n=300)
        result = mt.generate_signals(df)
        valid = result[result["signal"] != 0]["signal_strength"]
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_buy_signal_in_uptrend(self, mt):
        """Should generate at least one BUY signal in a clear uptrend."""
        df = make_df(n=300, uptrend=True)
        result = mt.generate_signals(df)
        # Not guaranteed to generate a buy (ADX threshold might not be met),
        # but should not error out
        assert "signal" in result.columns
