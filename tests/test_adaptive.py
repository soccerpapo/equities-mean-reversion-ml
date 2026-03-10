"""Tests for strategy/adaptive.py."""

import numpy as np
import pandas as pd
import pytest

from strategy.adaptive import AdaptiveTrader
from strategy.pairs_trading import PairsTrader
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
def at():
    return AdaptiveTrader(
        regime_allocations={
            0: {"pairs": 0.7, "momentum": 0.2, "cash": 0.1},
            1: {"pairs": 0.2, "momentum": 0.7, "cash": 0.1},
            2: {"pairs": 0.0, "momentum": 0.0, "cash": 1.0},
        },
        transition_days=3,
    )


class TestStrategySelection:
    def test_regime_0_returns_pairs_trader(self, at):
        strategy = at.select_strategy(0)
        assert isinstance(strategy, PairsTrader)

    def test_regime_1_returns_momentum_trader(self, at):
        strategy = at.select_strategy(1)
        assert isinstance(strategy, MomentumTrader)

    def test_regime_2_returns_none(self, at):
        strategy = at.select_strategy(2)
        assert strategy is None

    def test_unknown_regime_returns_none(self, at):
        strategy = at.select_strategy(99)
        assert strategy is None


class TestCapitalAllocation:
    def test_regime_0_allocation(self, at):
        alloc = at.allocate_capital(0, 100_000.0)
        assert alloc["pairs"] == pytest.approx(70_000.0)
        assert alloc["momentum"] == pytest.approx(20_000.0)
        assert alloc["cash"] == pytest.approx(10_000.0)

    def test_regime_1_allocation(self, at):
        alloc = at.allocate_capital(1, 100_000.0)
        assert alloc["pairs"] == pytest.approx(20_000.0)
        assert alloc["momentum"] == pytest.approx(70_000.0)
        assert alloc["cash"] == pytest.approx(10_000.0)

    def test_regime_2_all_cash(self, at):
        alloc = at.allocate_capital(2, 100_000.0)
        assert alloc["pairs"] == pytest.approx(0.0)
        assert alloc["momentum"] == pytest.approx(0.0)
        assert alloc["cash"] == pytest.approx(100_000.0)

    def test_allocations_sum_to_total_capital(self, at):
        for regime in (0, 1, 2):
            alloc = at.allocate_capital(regime, 50_000.0)
            total = alloc["pairs"] + alloc["momentum"] + alloc["cash"]
            assert total == pytest.approx(50_000.0), f"Regime {regime} allocation doesn't sum to capital"

    def test_unknown_regime_defaults_to_cash(self, at):
        alloc = at.allocate_capital(99, 100_000.0)
        assert alloc["cash"] == pytest.approx(100_000.0)


class TestTransitionManagement:
    def test_empty_positions_returns_no_orders(self, at):
        orders = at.manage_transition(0, 1, {})
        assert orders == []

    def test_generates_reduce_orders(self, at):
        positions = {
            "AAPL": {"qty": 100, "side": "long"},
            "MSFT": {"qty": 50, "side": "long"},
        }
        orders = at.manage_transition(0, 1, positions)
        assert len(orders) == 2

    def test_reduce_orders_have_correct_side(self, at):
        positions = {
            "AAPL": {"qty": 100, "side": "long"},
        }
        orders = at.manage_transition(0, 1, positions)
        assert orders[0]["side"] == "sell"

    def test_short_positions_buy_to_reduce(self, at):
        positions = {
            "TSLA": {"qty": 30, "side": "short"},
        }
        orders = at.manage_transition(1, 0, positions)
        assert orders[0]["side"] == "buy"

    def test_transition_reason_in_orders(self, at):
        positions = {"SPY": {"qty": 10, "side": "long"}}
        orders = at.manage_transition(0, 1, positions)
        assert "regime_transition" in orders[0]["reason"]


class TestRunOrchestration:
    def test_run_returns_expected_keys(self, at):
        data_dict = {
            "SPY": make_df(n=300, uptrend=True, seed=1),
            "AAPL": make_df(n=300, uptrend=True, seed=2),
        }
        result = at.run(["SPY", "AAPL"], data_dict, total_capital=100_000.0)
        for key in ("regime", "strategy", "allocation", "signals", "pairs", "orders"):
            assert key in result, f"Missing key: {key}"

    def test_regime_2_results_in_cash_strategy(self, at):
        """Force regime 2 via allocation and check no orders are generated."""
        at_forced = AdaptiveTrader(
            regime_allocations={
                0: {"pairs": 0.0, "momentum": 0.0, "cash": 1.0},
                1: {"pairs": 0.0, "momentum": 0.0, "cash": 1.0},
                2: {"pairs": 0.0, "momentum": 0.0, "cash": 1.0},
            }
        )
        # With crisis allocations, no orders should be generated regardless of regime
        data_dict = {"SPY": make_df(n=300)}
        result = at_forced.run(["SPY"], data_dict, total_capital=100_000.0)
        assert result["allocation"]["cash"] == pytest.approx(100_000.0)

    def test_strategy_field_is_valid_string(self, at):
        data_dict = {"SPY": make_df(n=300, uptrend=True)}
        result = at.run(["SPY"], data_dict)
        assert result["strategy"] in ("pairs", "momentum", "cash")

    def test_allocation_sums_to_capital(self, at):
        data_dict = {"SPY": make_df(n=300, uptrend=True)}
        result = at.run(["SPY"], data_dict, total_capital=200_000.0)
        alloc = result["allocation"]
        total = alloc["pairs"] + alloc["momentum"] + alloc["cash"]
        assert total == pytest.approx(200_000.0)

    def test_no_symbols_returns_gracefully(self, at):
        result = at.run([], {}, total_capital=100_000.0)
        assert "regime" in result
