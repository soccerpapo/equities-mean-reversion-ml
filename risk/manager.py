import logging
import numpy as np
import pandas as pd
from config import settings

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages position sizing and risk controls."""

    def calculate_atr_stops(
        self,
        entry_price: float,
        atr: float,
        side: str,
        atr_stop_mult: float = 2.0,
        atr_profit_mult: float = 3.0,
    ) -> tuple:
        """Calculate dynamic stop-loss and take-profit prices based on ATR.

        Args:
            entry_price: Price at trade entry
            atr: Average True Range value at entry
            side: 'long' or 'short'
            atr_stop_mult: Multiplier for stop-loss distance (default 2.0)
            atr_profit_mult: Multiplier for take-profit distance (default 3.0)

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if entry_price <= 0 or atr <= 0:
            return (0.0, 0.0)
        if side == "long":
            stop_loss = entry_price - atr_stop_mult * atr
            take_profit = entry_price + atr_profit_mult * atr
        else:  # short
            stop_loss = entry_price + atr_stop_mult * atr
            take_profit = entry_price - atr_profit_mult * atr
        return (stop_loss, take_profit)

    def apply_regime_sizing(self, base_size: int, regime_multiplier: float) -> int:
        """Scale position size by a regime multiplier.

        Args:
            base_size: Base number of shares from normal position sizing
            regime_multiplier: Multiplier from regime detector (0.0, 0.5, or 1.0)

        Returns:
            Adjusted number of shares (floored to int)
        """
        return int(base_size * max(0.0, regime_multiplier))

    def calculate_position_size(
        self,
        account_value: float,
        price: float,
        signal_strength: float,
        regime_multiplier: float = 1.0,
    ) -> int:
        """Calculate number of shares using fixed fractional sizing.

        Args:
            account_value: Total portfolio value
            price: Current asset price
            signal_strength: Signal confidence (0-1)
            regime_multiplier: Position size multiplier from regime detector (default 1.0)

        Returns:
            Number of shares to trade
        """
        if price <= 0 or account_value <= 0:
            return 0
        max_dollar = account_value * settings.MAX_POSITION_SIZE_PCT
        adjusted_dollar = max_dollar * max(0.0, min(1.0, signal_strength))
        shares = int(adjusted_dollar / price)
        return self.apply_regime_sizing(shares, regime_multiplier)

    def check_stop_loss(self, entry_price: float, current_price: float, side: str) -> bool:
        """Check if stop loss has been triggered.

        Args:
            entry_price: Price at trade entry
            current_price: Current market price
            side: 'long' or 'short'

        Returns:
            True if stop loss triggered
        """
        if entry_price <= 0:
            return False
        if side == "long":
            return current_price <= entry_price * (1 - settings.STOP_LOSS_PCT)
        elif side == "short":
            return current_price >= entry_price * (1 + settings.STOP_LOSS_PCT)
        return False

    def check_take_profit(self, entry_price: float, current_price: float, side: str) -> bool:
        """Check if take profit has been triggered.

        Args:
            entry_price: Price at trade entry
            current_price: Current market price
            side: 'long' or 'short'

        Returns:
            True if take profit triggered
        """
        if entry_price <= 0:
            return False
        if side == "long":
            return current_price >= entry_price * (1 + settings.TAKE_PROFIT_PCT)
        elif side == "short":
            return current_price <= entry_price * (1 - settings.TAKE_PROFIT_PCT)
        return False

    def check_max_drawdown(self, peak_value: float, current_value: float) -> bool:
        """Check if max portfolio drawdown has been exceeded.

        Args:
            peak_value: Portfolio peak value
            current_value: Current portfolio value

        Returns:
            True if max drawdown exceeded
        """
        if peak_value <= 0:
            return False
        drawdown = (peak_value - current_value) / peak_value
        return drawdown > settings.MAX_PORTFOLIO_DRAWDOWN_PCT

    def check_position_limits(self, current_positions: int, max_positions: int = 5) -> bool:
        """Check if position limit has been reached.

        Args:
            current_positions: Number of open positions
            max_positions: Maximum allowed positions

        Returns:
            True if at or above limit
        """
        return current_positions >= max_positions

    def get_risk_metrics(self, trades_df: pd.DataFrame) -> dict:
        """Calculate portfolio risk metrics.

        Args:
            trades_df: DataFrame with 'pnl' and 'return' columns

        Returns:
            Dict with Sharpe, Sortino, max drawdown, win rate, profit factor
        """
        if trades_df.empty or "pnl" not in trades_df.columns:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }

        returns = trades_df["pnl"]
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
        # None indicates no losing trades (perfect record); callers should handle this sentinel
        profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else None

        mean_return = returns.mean()
        std_return = returns.std()
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return != 0 else 0.0

        downside = returns[returns < 0].std()
        sortino = (mean_return / downside * np.sqrt(252)) if downside != 0 else 0.0

        cumulative = returns.cumsum()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / (running_max.abs() + 1e-9)
        max_dd = drawdown.max()

        return {
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(float(max_dd), 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        }
