import logging
import os
from typing import Dict, List, Optional
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
        self._benchmark: Optional[pd.Series] = None
        self._regimes: Optional[pd.Series] = None  # regime label per day

    def run(
        self,
        df: pd.DataFrame,
        signals_df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        use_atr_stops: bool = False,
        atr_stop_mult: float = 2.0,
        atr_profit_mult: float = 3.0,
        regimes: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Simulate trades from signals DataFrame.

        Args:
            df: OHLCV DataFrame (price data)
            signals_df: DataFrame with 'signal' and 'signal_strength' columns
            initial_capital: Starting portfolio value
            use_atr_stops: Whether to use ATR-based dynamic stops instead of
                           fixed percentage stops
            atr_stop_mult: ATR multiplier for stop-loss (used when use_atr_stops=True)
            atr_profit_mult: ATR multiplier for take-profit (used when use_atr_stops=True)
            regimes: Optional Series of regime labels per date for logging/plotting

        Returns:
            DataFrame with portfolio equity curve
        """
        from config import settings

        self._initial_capital = initial_capital
        self._trades = []
        self._regimes = regimes

        cash = initial_capital
        position = 0
        entry_price = 0.0
        entry_date = None
        stop_loss_price = 0.0
        take_profit_price = 0.0
        portfolio_values = []

        # Compute buy-and-hold benchmark using Close prices
        close_prices = signals_df["Close"] if "Close" in signals_df.columns else df["Close"]
        first_price = close_prices.iloc[0]
        if first_price and first_price != 0:
            benchmark_shares = initial_capital / first_price
            self._benchmark = close_prices * benchmark_shares
        else:
            self._benchmark = None

        for idx, row in signals_df.iterrows():
            price = row["Close"] if "Close" in row.index else df.loc[idx, "Close"]
            signal = int(row.get("signal", 0))
            strength = float(row.get("signal_strength", 0.5))
            atr = float(row.get("atr", 0.0)) if "atr" in row.index else 0.0

            exec_price = price * (1 + self.SLIPPAGE) if signal == 1 else price * (1 - self.SLIPPAGE)

            # Check ATR or fixed stops for open positions
            if position != 0:
                exit_by_stop = False
                if use_atr_stops and stop_loss_price > 0:
                    if position > 0 and price <= stop_loss_price:
                        exit_by_stop = True
                    elif position < 0 and price >= stop_loss_price:
                        exit_by_stop = True
                elif not use_atr_stops:
                    if position > 0 and price <= entry_price * (1 - settings.STOP_LOSS_PCT):
                        exit_by_stop = True
                    elif position < 0 and price >= entry_price * (1 + settings.STOP_LOSS_PCT):
                        exit_by_stop = True

                exit_by_tp = False
                if use_atr_stops and take_profit_price > 0:
                    if position > 0 and price >= take_profit_price:
                        exit_by_tp = True
                    elif position < 0 and price <= take_profit_price:
                        exit_by_tp = True
                elif not use_atr_stops:
                    if position > 0 and price >= entry_price * (1 + settings.TAKE_PROFIT_PCT):
                        exit_by_tp = True
                    elif position < 0 and price <= entry_price * (1 - settings.TAKE_PROFIT_PCT):
                        exit_by_tp = True

                exit_by_signal = (position > 0 and signal == -1) or (position < 0 and signal == 1)

                if exit_by_stop or exit_by_tp or exit_by_signal:
                    exit_exec = price * (1 - self.SLIPPAGE) if position > 0 else price * (1 + self.SLIPPAGE)
                    pnl = (exit_exec - entry_price) * position
                    cash += exit_exec * abs(position)
                    self._trades.append({
                        "entry_date": entry_date,
                        "exit_date": idx,
                        "symbol": "ASSET",
                        "side": "long" if position > 0 else "short",
                        "entry_price": entry_price,
                        "exit_price": exit_exec,
                        "qty": abs(position),
                        "pnl": pnl,
                        "exit_reason": "stop" if exit_by_stop else ("tp" if exit_by_tp else "signal"),
                    })
                    position = 0
                    stop_loss_price = 0.0
                    take_profit_price = 0.0

            # Enter new position
            if signal != 0 and position == 0:
                max_invest = cash * 0.1 * strength
                shares = int(max_invest / exec_price)
                if shares > 0:
                    position = shares if signal == 1 else -shares
                    entry_price = exec_price
                    entry_date = idx
                    cash -= exec_price * abs(position)
                    # Compute ATR-based stops if requested
                    if use_atr_stops and atr > 0:
                        if signal == 1:
                            stop_loss_price = entry_price - atr_stop_mult * atr
                            take_profit_price = entry_price + atr_profit_mult * atr
                        else:
                            stop_loss_price = entry_price + atr_stop_mult * atr
                            take_profit_price = entry_price - atr_profit_mult * atr

            port_val = cash + position * price
            portfolio_values.append({"date": idx, "portfolio_value": port_val, "cash": cash})

        self._portfolio = pd.DataFrame(portfolio_values).set_index("date")
        return self._portfolio

    def get_performance_report(self) -> Dict:
        """Return comprehensive performance metrics including benchmark comparison.

        Returns:
            Dict with return, Sharpe, Sortino, drawdown, win/loss stats, and alpha
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

        # Benchmark (buy-and-hold) metrics
        benchmark_return = float("nan")
        alpha = float("nan")
        if self._benchmark is not None and len(self._benchmark) > 0:
            benchmark_return = (self._benchmark.iloc[-1] / self._initial_capital) - 1
            alpha = total_return - benchmark_return

        report = {
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
        if not np.isnan(benchmark_return):
            report["benchmark_return"] = round(float(benchmark_return), 4)
            report["alpha"] = round(float(alpha), 4)
        return report

    def plot_results(self, output_dir: str = ".", regime_series: Optional[pd.Series] = None) -> None:
        """Generate and save equity curve and drawdown charts with benchmark.

        Args:
            output_dir: Directory to save chart files
            regime_series: Optional Series with regime labels (0/1/2) indexed by date.
                           When provided, the equity-curve panel is shaded by regime.
        """
        if self._portfolio is None or self._portfolio.empty:
            logger.warning("No portfolio data to plot")
            return

        os.makedirs(output_dir, exist_ok=True)
        values = self._portfolio["portfolio_value"]
        running_max = values.cummax()
        drawdown = (values - running_max) / running_max * 100

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # ---- Equity curve with optional regime shading ----
        ax = axes[0]
        ax.plot(values.index, values, label="Strategy", color="blue")
        ax.axhline(self._initial_capital, color="gray", linestyle="--", label="Initial Capital")
        if self._benchmark is not None:
            ax.plot(self._benchmark.index, self._benchmark, label="Buy & Hold", color="orange", alpha=0.7)

        # Regime shading: green=0 (low-vol / mean-reverting), yellow=1 (normal), red=2 (crisis)
        regime_colors = {0: "#d4edda", 1: "#fff3cd", 2: "#f8d7da"}
        regime_labels = {0: "Regime 0 (Low-vol)", 1: "Regime 1 (Normal)", 2: "Regime 2 (High-vol)"}
        rs = regime_series if regime_series is not None else self._regimes
        if rs is not None and len(rs) > 0:
            regime_aligned = rs.reindex(values.index).ffill()
            prev_regime = None
            start_date = None
            added_labels: set = set()
            for date, regime in regime_aligned.items():
                if pd.isna(regime):
                    continue
                regime = int(regime)
                if prev_regime is None:
                    prev_regime = regime
                    start_date = date
                elif regime != prev_regime:
                    color = regime_colors.get(prev_regime, "white")
                    lbl = regime_labels[prev_regime] if prev_regime not in added_labels else None
                    ax.axvspan(start_date, date, alpha=0.3, color=color, label=lbl)
                    if lbl:
                        added_labels.add(prev_regime)
                    prev_regime = regime
                    start_date = date
            # Last segment
            if start_date is not None and prev_regime is not None:
                color = regime_colors.get(prev_regime, "white")
                lbl = regime_labels[prev_regime] if prev_regime not in added_labels else None
                ax.axvspan(start_date, values.index[-1], alpha=0.3, color=color, label=lbl)

        ax.set_title("Equity Curve")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

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

    @staticmethod
    def print_comparison_table(rows: List[Dict]) -> None:
        """Print a formatted comparison table of multiple strategy runs.

        Args:
            rows: List of dicts, each with keys: 'approach', 'total_return',
                  'sharpe_ratio', 'max_drawdown', 'num_trades', 'win_rate'.
        """
        cols = ["approach", "total_return", "sharpe_ratio", "max_drawdown", "num_trades", "win_rate"]
        headers = ["Approach", "Return", "Sharpe", "MaxDD", "Trades", "WinRate"]
        col_w = [24, 10, 10, 10, 8, 10]

        header_line = " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
        sep = "-+-".join("-" * w for w in col_w)

        print("\n=== Strategy Comparison ===")
        print(header_line)
        print(sep)
        for row in rows:
            vals = []
            for i, col in enumerate(cols):
                v = row.get(col, "N/A")
                if col in ("total_return", "max_drawdown"):
                    v = f"{float(v)*100:.2f}%" if v != "N/A" else "N/A"
                elif col == "sharpe_ratio":
                    v = f"{float(v):.2f}" if v != "N/A" else "—"
                elif col == "win_rate":
                    v = f"{float(v)*100:.1f}%" if v != "N/A" else "—"
                else:
                    v = str(v)
                vals.append(v.ljust(col_w[i]))
            print(" | ".join(vals))
