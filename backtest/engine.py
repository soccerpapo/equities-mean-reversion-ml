import logging
import os
from typing import Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Event-driven backtesting engine with slippage modeling."""

    SLIPPAGE = 0.0001  # 0.01%

    def __init__(self):
        self._portfolio: Optional[pd.DataFrame] = None
        self._trades: list = []
        self._initial_capital: float = 100_000.0

    def run(
        self, df: pd.DataFrame, signals_df: pd.DataFrame, initial_capital: float = 100_000.0
    ) -> pd.DataFrame:
        """Simulate trades from signals DataFrame.

        Args:
            df: OHLCV DataFrame (price data)
            signals_df: DataFrame with 'signal' and 'signal_strength' columns
            initial_capital: Starting portfolio value

        Returns:
            DataFrame with portfolio equity curve
        """
        self._initial_capital = initial_capital
        self._trades = []

        cash = initial_capital
        position = 0
        entry_price = 0.0
        entry_date = None
        portfolio_values = []

        for idx, row in signals_df.iterrows():
            price = row["Close"] if "Close" in row.index else df.loc[idx, "Close"]
            signal = int(row.get("signal", 0))
            strength = float(row.get("signal_strength", 0.5))

            exec_price = price * (1 + self.SLIPPAGE) if signal == 1 else price * (1 - self.SLIPPAGE)

            # Exit existing position
            if position != 0:
                if (position > 0 and signal == -1) or (position < 0 and signal == 1):
                    pnl = (exec_price - entry_price) * position
                    cash += exec_price * abs(position)
                    self._trades.append({
                        "entry_date": entry_date,
                        "exit_date": idx,
                        "symbol": "ASSET",
                        "side": "long" if position > 0 else "short",
                        "entry_price": entry_price,
                        "exit_price": exec_price,
                        "qty": abs(position),
                        "pnl": pnl,
                    })
                    position = 0

            # Enter new position
            if signal != 0 and position == 0:
                max_invest = cash * 0.1 * strength
                shares = int(max_invest / exec_price)
                if shares > 0:
                    position = shares if signal == 1 else -shares
                    entry_price = exec_price
                    entry_date = idx
                    cash -= exec_price * abs(position)

            port_val = cash + position * price
            portfolio_values.append({"date": idx, "portfolio_value": port_val, "cash": cash})

        self._portfolio = pd.DataFrame(portfolio_values).set_index("date")
        return self._portfolio

    def get_performance_report(self) -> Dict:
        """Return comprehensive performance metrics.

        Returns:
            Dict with return, Sharpe, Sortino, drawdown, win/loss stats
        """
        if self._portfolio is None or self._portfolio.empty:
            return {}

        values = self._portfolio["portfolio_value"]
        returns = values.pct_change().dropna()

        total_return = (values.iloc[-1] / self._initial_capital) - 1
        n_years = len(returns) / 252
        # Require at least ~25 trading days to produce a meaningful annualized figure
        MIN_YEARS_FOR_ANNUALIZATION = 0.1
        if n_years >= MIN_YEARS_FOR_ANNUALIZATION:
            ann_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            ann_return = float("nan")

        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0.0
        downside = returns[returns < 0].std()
        sortino = (returns.mean() / downside * np.sqrt(252)) if downside != 0 else 0.0

        running_max = values.cummax()
        drawdown = (values - running_max) / running_max
        max_drawdown = drawdown.min()

        trades_df = pd.DataFrame(self._trades)
        num_trades = len(trades_df)

        if num_trades > 0:
            wins = trades_df[trades_df["pnl"] > 0]
            losses = trades_df[trades_df["pnl"] <= 0]
            win_rate = len(wins) / num_trades
            avg_win = wins["pnl"].mean() if len(wins) > 0 else 0.0
            avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0.0
            # None indicates no losing trades (perfect record); callers should handle this sentinel
            profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())) if losses["pnl"].sum() != 0 else None

            outcomes = (trades_df["pnl"] > 0).astype(int).tolist()
            max_win_streak = max_loss_streak = cur_win = cur_loss = 0
            for o in outcomes:
                if o == 1:
                    cur_win += 1
                    cur_loss = 0
                else:
                    cur_loss += 1
                    cur_win = 0
                max_win_streak = max(max_win_streak, cur_win)
                max_loss_streak = max(max_loss_streak, cur_loss)
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0
            max_win_streak = max_loss_streak = 0

        return {
            "total_return": round(total_return, 4),
            "annualized_return": round(ann_return, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
            "num_trades": num_trades,
            "longest_win_streak": max_win_streak,
            "longest_loss_streak": max_loss_streak,
        }

    def plot_results(self, output_dir: str = ".") -> None:
        """Generate and save equity curve and drawdown charts.

        Args:
            output_dir: Directory to save chart files
        """
        if self._portfolio is None or self._portfolio.empty:
            logger.warning("No portfolio data to plot")
            return

        os.makedirs(output_dir, exist_ok=True)
        values = self._portfolio["portfolio_value"]
        running_max = values.cummax()
        drawdown = (values - running_max) / running_max * 100

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(values.index, values, label="Portfolio Value", color="blue")
        axes[0].axhline(self._initial_capital, color="gray", linestyle="--", label="Initial Capital")
        axes[0].set_title("Equity Curve")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.4, label="Drawdown")
        axes[1].set_title("Drawdown (%)")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "backtest_results.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Plot saved to {path}")
