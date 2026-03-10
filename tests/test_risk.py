import numpy as np
import pandas as pd
import pytest
from risk.manager import RiskManager


@pytest.fixture
def risk_mgr():
    return RiskManager()


class TestPositionSizing:
    def test_basic_position_size(self, risk_mgr):
        qty = risk_mgr.calculate_position_size(100_000, 100.0, 1.0)
        # 10% of 100k = 10k; 10k / 100 = 100 shares
        assert qty == 100

    def test_position_size_scales_with_strength(self, risk_mgr):
        qty_full = risk_mgr.calculate_position_size(100_000, 100.0, 1.0)
        qty_half = risk_mgr.calculate_position_size(100_000, 100.0, 0.5)
        assert qty_half < qty_full

    def test_zero_price_returns_zero(self, risk_mgr):
        qty = risk_mgr.calculate_position_size(100_000, 0.0, 1.0)
        assert qty == 0

    def test_zero_strength_returns_zero(self, risk_mgr):
        qty = risk_mgr.calculate_position_size(100_000, 100.0, 0.0)
        assert qty == 0


class TestStopLoss:
    def test_long_stop_loss_triggered(self, risk_mgr):
        # Entry 100, stop at 2% = 98. Price 97 should trigger.
        assert risk_mgr.check_stop_loss(100.0, 97.0, "long") is True

    def test_long_stop_loss_not_triggered(self, risk_mgr):
        assert risk_mgr.check_stop_loss(100.0, 99.5, "long") is False

    def test_short_stop_loss_triggered(self, risk_mgr):
        assert risk_mgr.check_stop_loss(100.0, 103.0, "short") is True

    def test_short_stop_loss_not_triggered(self, risk_mgr):
        assert risk_mgr.check_stop_loss(100.0, 100.5, "short") is False


class TestTakeProfit:
    def test_long_take_profit_triggered(self, risk_mgr):
        # Entry 100, take profit at 4% = 104. Price 105 should trigger.
        assert risk_mgr.check_take_profit(100.0, 105.0, "long") is True

    def test_long_take_profit_not_triggered(self, risk_mgr):
        assert risk_mgr.check_take_profit(100.0, 102.0, "long") is False

    def test_short_take_profit_triggered(self, risk_mgr):
        assert risk_mgr.check_take_profit(100.0, 95.0, "short") is True

    def test_short_take_profit_not_triggered(self, risk_mgr):
        assert risk_mgr.check_take_profit(100.0, 98.0, "short") is False


class TestMaxDrawdown:
    def test_drawdown_exceeded(self, risk_mgr):
        # Peak 100k, current 85k => 15% drawdown > 10% threshold
        assert risk_mgr.check_max_drawdown(100_000, 85_000) is True

    def test_drawdown_not_exceeded(self, risk_mgr):
        # Peak 100k, current 95k => 5% drawdown
        assert risk_mgr.check_max_drawdown(100_000, 95_000) is False

    def test_zero_peak_returns_false(self, risk_mgr):
        assert risk_mgr.check_max_drawdown(0, 50_000) is False


class TestPositionLimits:
    def test_at_limit(self, risk_mgr):
        assert risk_mgr.check_position_limits(5, max_positions=5) is True

    def test_over_limit(self, risk_mgr):
        assert risk_mgr.check_position_limits(6, max_positions=5) is True

    def test_under_limit(self, risk_mgr):
        assert risk_mgr.check_position_limits(3, max_positions=5) is False
