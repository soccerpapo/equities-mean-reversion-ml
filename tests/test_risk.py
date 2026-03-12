import numpy as np
import pandas as pd
import pytest
from risk.manager import RiskManager
from config import settings


@pytest.fixture
def risk_mgr():
    return RiskManager()


class TestPositionSizing:
    def test_basic_position_size(self, risk_mgr):
        qty = risk_mgr.calculate_position_size(100_000, 100.0, 1.0)
        expected = int(100_000 * settings.MAX_POSITION_SIZE_PCT / 100.0)
        assert qty == expected

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

    def test_regime_multiplier_scales_size(self, risk_mgr):
        qty_full = risk_mgr.calculate_position_size(100_000, 100.0, 1.0, regime_multiplier=1.0)
        qty_half = risk_mgr.calculate_position_size(100_000, 100.0, 1.0, regime_multiplier=0.5)
        qty_zero = risk_mgr.calculate_position_size(100_000, 100.0, 1.0, regime_multiplier=0.0)
        assert qty_half < qty_full
        assert qty_zero == 0


class TestStopLoss:
    def test_long_stop_loss_triggered(self, risk_mgr):
        stop_pct = settings.STOP_LOSS_PCT
        trigger_price = 100.0 * (1 - stop_pct) - 1.0
        assert risk_mgr.check_stop_loss(100.0, trigger_price, "long") is True

    def test_long_stop_loss_not_triggered(self, risk_mgr):
        safe_price = 100.0 * (1 - settings.STOP_LOSS_PCT) + 1.0
        assert risk_mgr.check_stop_loss(100.0, safe_price, "long") is False

    def test_short_stop_loss_triggered(self, risk_mgr):
        trigger_price = 100.0 * (1 + settings.STOP_LOSS_PCT) + 1.0
        assert risk_mgr.check_stop_loss(100.0, trigger_price, "short") is True

    def test_short_stop_loss_not_triggered(self, risk_mgr):
        safe_price = 100.0 * (1 + settings.STOP_LOSS_PCT) - 1.0
        assert risk_mgr.check_stop_loss(100.0, safe_price, "short") is False


class TestTakeProfit:
    def test_long_take_profit_triggered(self, risk_mgr):
        trigger_price = 100.0 * (1 + settings.TAKE_PROFIT_PCT) + 1.0
        assert risk_mgr.check_take_profit(100.0, trigger_price, "long") is True

    def test_long_take_profit_not_triggered(self, risk_mgr):
        safe_price = 100.0 * (1 + settings.TAKE_PROFIT_PCT) - 1.0
        assert risk_mgr.check_take_profit(100.0, safe_price, "long") is False

    def test_short_take_profit_triggered(self, risk_mgr):
        trigger_price = 100.0 * (1 - settings.TAKE_PROFIT_PCT) - 1.0
        assert risk_mgr.check_take_profit(100.0, trigger_price, "short") is True

    def test_short_take_profit_not_triggered(self, risk_mgr):
        safe_price = 100.0 * (1 - settings.TAKE_PROFIT_PCT) + 1.0
        assert risk_mgr.check_take_profit(100.0, safe_price, "short") is False


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


class TestATRStops:
    def test_long_stop_is_below_entry(self, risk_mgr):
        sl, tp = risk_mgr.calculate_atr_stops(100.0, 2.0, "long", 2.0, 3.0)
        # stop = 100 - 2*2 = 96
        assert sl == pytest.approx(96.0)

    def test_long_tp_is_above_entry(self, risk_mgr):
        sl, tp = risk_mgr.calculate_atr_stops(100.0, 2.0, "long", 2.0, 3.0)
        # tp = 100 + 3*2 = 106
        assert tp == pytest.approx(106.0)

    def test_short_stop_is_above_entry(self, risk_mgr):
        sl, tp = risk_mgr.calculate_atr_stops(100.0, 2.0, "short", 2.0, 3.0)
        # stop = 100 + 2*2 = 104
        assert sl == pytest.approx(104.0)

    def test_short_tp_is_below_entry(self, risk_mgr):
        sl, tp = risk_mgr.calculate_atr_stops(100.0, 2.0, "short", 2.0, 3.0)
        # tp = 100 - 3*2 = 94
        assert tp == pytest.approx(94.0)

    def test_zero_price_returns_zero_pair(self, risk_mgr):
        sl, tp = risk_mgr.calculate_atr_stops(0.0, 2.0, "long")
        assert sl == 0.0
        assert tp == 0.0

    def test_custom_multipliers(self, risk_mgr):
        sl, tp = risk_mgr.calculate_atr_stops(200.0, 5.0, "long", atr_stop_mult=1.5, atr_profit_mult=4.0)
        assert sl == pytest.approx(200.0 - 1.5 * 5.0)
        assert tp == pytest.approx(200.0 + 4.0 * 5.0)


class TestRegimeSizing:
    def test_full_multiplier(self, risk_mgr):
        assert risk_mgr.apply_regime_sizing(100, 1.0) == 100

    def test_half_multiplier(self, risk_mgr):
        assert risk_mgr.apply_regime_sizing(100, 0.5) == 50

    def test_zero_multiplier(self, risk_mgr):
        assert risk_mgr.apply_regime_sizing(100, 0.0) == 0

    def test_negative_multiplier_clamps_to_zero(self, risk_mgr):
        assert risk_mgr.apply_regime_sizing(100, -0.5) == 0

