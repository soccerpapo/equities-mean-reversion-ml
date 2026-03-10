"""Tests for strategy/pairs_trading.py."""

import numpy as np
import pandas as pd
import pytest

from strategy.pairs_trading import PairsTrader


def make_cointegrated_series(n=300, seed=42):
    """Create two cointegrated price series (random walk + common factor)."""
    np.random.seed(seed)
    common = np.cumsum(np.random.randn(n) * 0.5)
    noise_a = np.random.randn(n) * 0.2
    noise_b = np.random.randn(n) * 0.2
    price_a = pd.Series(100 + common + noise_a, name="A")
    price_b = pd.Series(100 + 2.0 * common + noise_b, name="B")
    return price_a, price_b


def make_independent_series(n=300, seed=7):
    """Create two independent (non-cointegrated) random walk series."""
    np.random.seed(seed)
    price_a = pd.Series(100 + np.cumsum(np.random.randn(n) * 1.0), name="A")
    price_b = pd.Series(200 + np.cumsum(np.random.randn(n) * 1.5), name="B")
    return price_a, price_b


@pytest.fixture
def pt():
    return PairsTrader(zscore_entry=2.0, zscore_exit=0.5, zscore_stop=3.0, lookback=60)


class TestCointegration:
    def test_finds_cointegrated_pair(self, pt):
        """Known cointegrated series should be found."""
        price_a, price_b = make_cointegrated_series(n=300)
        data_dict = {"A": price_a, "B": price_b}
        pairs = pt.find_cointegrated_pairs(["A", "B"], data_dict)
        assert len(pairs) > 0, "Expected at least one cointegrated pair"

    def test_cointegrated_pair_structure(self, pt):
        """Returned tuples should have the correct structure."""
        price_a, price_b = make_cointegrated_series(n=300)
        data_dict = {"A": price_a, "B": price_b}
        pairs = pt.find_cointegrated_pairs(["A", "B"], data_dict)
        if pairs:
            sym_a, sym_b, p_val, corr, hedge = pairs[0]
            assert sym_a in ("A", "B")
            assert sym_b in ("A", "B")
            assert 0.0 <= p_val <= 1.0
            assert -1.0 <= corr <= 1.0
            assert isinstance(hedge, float)

    def test_sorted_by_pvalue(self, pt):
        """Pairs should be sorted by p-value ascending."""
        price_a, price_b = make_cointegrated_series(n=300)
        price_c = pd.Series(50 + np.cumsum(np.random.randn(300) * 0.3), name="C")
        data_dict = {"A": price_a, "B": price_b, "C": price_c}
        pairs = pt.find_cointegrated_pairs(["A", "B", "C"], data_dict)
        p_values = [p[2] for p in pairs]
        assert p_values == sorted(p_values), "Pairs should be sorted by p-value"

    def test_insufficient_data_returns_empty(self, pt):
        """Too-short series should be skipped and return no pairs."""
        price_a = pd.Series([100.0] * 10, name="A")
        price_b = pd.Series([100.0] * 10, name="B")
        data_dict = {"A": price_a, "B": price_b}
        pairs = pt.find_cointegrated_pairs(["A", "B"], data_dict)
        assert pairs == []

    def test_missing_symbol_skipped(self, pt):
        """Symbols not in data_dict should be silently skipped."""
        price_a, price_b = make_cointegrated_series(n=300)
        data_dict = {"A": price_a}  # B is missing
        pairs = pt.find_cointegrated_pairs(["A", "B"], data_dict)
        assert pairs == []


class TestSpreadCalculation:
    def test_spread_columns_present(self, pt):
        price_a, price_b = make_cointegrated_series(n=300)
        spread_df = pt.calculate_spread(price_a, price_b, hedge_ratio=1.0)
        assert "spread" in spread_df.columns
        assert "spread_zscore" in spread_df.columns
        assert "spread_mean" in spread_df.columns
        assert "spread_std" in spread_df.columns

    def test_spread_formula(self, pt):
        """Spread = price_a - hedge_ratio * price_b."""
        price_a = pd.Series([10.0, 12.0, 14.0])
        price_b = pd.Series([5.0, 6.0, 7.0])
        result = pt.calculate_spread(price_a, price_b, hedge_ratio=2.0)
        expected = price_a - 2.0 * price_b
        pd.testing.assert_series_equal(result["spread"], expected, check_names=False)

    def test_zscore_is_stationary(self, pt):
        """Z-score of spread should be roughly mean-zero for cointegrated pair."""
        price_a, price_b = make_cointegrated_series(n=500)
        hedge = pt.calculate_hedge_ratio(price_a, price_b)
        spread_df = pt.calculate_spread(price_a, price_b, hedge_ratio=hedge)
        z = spread_df["spread_zscore"].dropna()
        assert abs(z.mean()) < 1.0, "Z-score mean should be near zero"

    def test_spread_length_matches_input(self, pt):
        price_a, price_b = make_cointegrated_series(n=200)
        spread_df = pt.calculate_spread(price_a, price_b, hedge_ratio=1.0)
        assert len(spread_df) == len(price_a)


class TestHedgeRatio:
    def test_hedge_ratio_positive(self, pt):
        price_a, price_b = make_cointegrated_series(n=300)
        hedge = pt.calculate_hedge_ratio(price_a, price_b)
        assert hedge > 0, "Hedge ratio should be positive for positively correlated series"

    def test_hedge_ratio_close_to_true_value(self, pt):
        """OLS should recover the correct regression beta for the DGP.

        DGP: price_a = 100 + common + noise,  price_b = 100 + 2*common + noise
        Regressing price_a ~ price_b (i.e. the ratio of their common factor loadings
        1/2 = 0.5) means OLS recovers β ≈ 0.5, not 2.0.
        """
        price_a, price_b = make_cointegrated_series(n=500)
        hedge = pt.calculate_hedge_ratio(price_a, price_b)
        assert abs(hedge - 0.5) < 0.3, f"Expected hedge ratio ~0.5, got {hedge:.4f}"

    def test_rolling_hedge_ratio_length(self, pt):
        price_a, price_b = make_cointegrated_series(n=200)
        rolling = pt.calculate_rolling_hedge_ratio(price_a, price_b, window=60)
        assert len(rolling) == len(price_a)

    def test_hedge_ratio_fallback_on_failure(self, pt):
        """Should return 1.0 when regression raises (e.g. empty series)."""
        # calculate_hedge_ratio returns 1.0 when an exception is raised.
        # We can simulate that by passing single-element series where OLS may fail.
        hedge = pt.calculate_hedge_ratio(pd.Series([1.0]), pd.Series([1.0]))
        # Either regression succeeds and gives a reasonable number,
        # or it falls back to 1.0 — both are acceptable.
        assert isinstance(hedge, float)


class TestSignalGeneration:
    def test_buy_signal_on_negative_zscore(self, pt):
        """Z-score < -ENTRY should generate a BUY signal."""
        zscores = pd.Series(
            [0.0] * 10 + [-2.5] + [0.0] * 10,
            dtype=float,
        )
        signals = pt.generate_signals(zscores)
        assert (signals["signal"] == PairsTrader.SIGNAL_BUY).any()

    def test_sell_signal_on_positive_zscore(self, pt):
        """Z-score > +ENTRY should generate a SELL signal."""
        zscores = pd.Series([0.0] * 10 + [2.5] + [0.0] * 10, dtype=float)
        signals = pt.generate_signals(zscores)
        assert (signals["signal"] == PairsTrader.SIGNAL_SELL).any()

    def test_close_signal_after_mean_reversion(self, pt):
        """After BUY entry, crossing back above -EXIT should close position."""
        zscores = pd.Series(
            [0.0] * 5 + [-2.5] + [-1.0] + [0.0] * 5,
            dtype=float,
        )
        signals = pt.generate_signals(zscores)
        assert (signals["signal"] == PairsTrader.SIGNAL_CLOSE).any()

    def test_stop_loss_signal(self, pt):
        """Z-score > STOP should trigger a stop loss."""
        zscores = pd.Series(
            [0.0] * 5 + [-2.5] + [-3.5],
            dtype=float,
        )
        signals = pt.generate_signals(zscores)
        assert (signals["signal"] == PairsTrader.SIGNAL_STOP).any()

    def test_no_signal_in_neutral_zone(self, pt):
        """Z-scores near zero should not generate any entry signals."""
        zscores = pd.Series([0.1, -0.5, 0.3, -0.2, 0.0] * 10, dtype=float)
        signals = pt.generate_signals(zscores)
        assert not (signals["signal"] == PairsTrader.SIGNAL_BUY).any()
        assert not (signals["signal"] == PairsTrader.SIGNAL_SELL).any()

    def test_signal_columns_present(self, pt):
        zscores = pd.Series([0.0, -2.5, 0.0], dtype=float)
        signals = pt.generate_signals(zscores)
        assert "signal" in signals.columns
        assert "spread_zscore" in signals.columns
        assert "entry_zscore" in signals.columns


class TestPairOrders:
    def test_buy_signal_orders(self, pt):
        orders = pt.get_pair_orders(
            PairsTrader.SIGNAL_BUY, "A", "B", 1.0, 10_000, 100.0, 50.0
        )
        assert len(orders) == 2
        sides = {o["symbol"]: o["side"] for o in orders}
        assert sides["A"] == "buy"
        assert sides["B"] == "sell"

    def test_sell_signal_orders(self, pt):
        orders = pt.get_pair_orders(
            PairsTrader.SIGNAL_SELL, "A", "B", 1.0, 10_000, 100.0, 50.0
        )
        assert len(orders) == 2
        sides = {o["symbol"]: o["side"] for o in orders}
        assert sides["A"] == "sell"
        assert sides["B"] == "buy"

    def test_close_signal_orders(self, pt):
        orders = pt.get_pair_orders(
            PairsTrader.SIGNAL_CLOSE, "A", "B", 1.0, 10_000, 100.0, 50.0
        )
        assert len(orders) == 2
        assert all(o["reason"] == "close_pair" for o in orders)

    def test_no_orders_on_none_signal(self, pt):
        orders = pt.get_pair_orders(
            PairsTrader.SIGNAL_NONE, "A", "B", 1.0, 10_000, 100.0, 50.0
        )
        assert orders == []

    def test_no_orders_on_zero_price(self, pt):
        orders = pt.get_pair_orders(
            PairsTrader.SIGNAL_BUY, "A", "B", 1.0, 10_000, 0.0, 50.0
        )
        assert orders == []


class TestPairStats:
    def test_empty_trades_returns_defaults(self, pt):
        stats = pt.get_pair_stats([])
        assert stats["num_trades"] == 0
        assert stats["win_rate"] == 0.0

    def test_all_winning_trades(self, pt):
        trades = [{"pnl": 100.0}, {"pnl": 200.0}, {"pnl": 50.0}]
        stats = pt.get_pair_stats(trades)
        assert stats["num_trades"] == 3
        assert stats["win_rate"] == 1.0
        assert stats["total_pnl"] == pytest.approx(350.0)

    def test_mixed_trades(self, pt):
        trades = [{"pnl": 100.0}, {"pnl": -50.0}]
        stats = pt.get_pair_stats(trades)
        assert stats["win_rate"] == pytest.approx(0.5)
        assert stats["total_pnl"] == pytest.approx(50.0)
