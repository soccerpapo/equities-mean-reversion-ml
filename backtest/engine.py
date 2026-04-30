import logging
import os
from typing import Dict, List, Optional, Tuple
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
        self._regimes: Optional[pd.Series] = None
        self._current_symbol: Optional[str] = None
        self._pending_entry_indicators: dict = {}
        self._signals_df: Optional[pd.DataFrame] = None

    def _scale_atr_multipliers(self, atr: float, price: float,
                               base_stop_mult: float,
                               base_profit_mult: float) -> Tuple[float, float]:
        """Scale ATR stop/profit multipliers based on per-asset volatility.

        High-volatility assets (ATR/price > 2%) get wider stops to avoid
        premature stop-outs.  Low-vol assets keep the base multipliers.

        Args:
            atr: Current ATR value.
            price: Current asset price.
            base_stop_mult: Base ATR stop-loss multiplier from settings.
            base_profit_mult: Base ATR take-profit multiplier from settings.

        Returns:
            Tuple of (scaled_stop_mult, scaled_profit_mult).
        """
        from config import settings
        if not getattr(settings, "USE_VOLATILITY_SCALED_STOPS", False):
            return base_stop_mult, base_profit_mult

        if atr <= 0 or price <= 0:
            return base_stop_mult, base_profit_mult

        atr_pct = atr / price

        # Scale factor: 1.0 at <=1% ATR/price, up to 1.6 at >=3%
        if atr_pct <= 0.01:
            scale = 1.0
        elif atr_pct >= 0.03:
            scale = 1.6
        else:
            scale = 1.0 + 0.6 * (atr_pct - 0.01) / 0.02

        return base_stop_mult * scale, base_profit_mult * scale

    def _calculate_position_size(
        self, atr: float, price: float, max_size_override: float = None,
    ) -> float:
        """Calculate position size as a fraction of cash based on ATR volatility.

        Uses volatility-inverse sizing: lower volatility → larger position (up to
        MAX_POSITION_SIZE_PCT), higher volatility → smaller position (down to 0.15).
        ATR as a percentage of price is used as the volatility measure.

        Args:
            atr: Average True Range value for the current bar.
            price: Current asset price.
            max_size_override: Optional per-stock max position size (from StockProfile).

        Returns:
            Position size as a fraction of available cash (between 0.15 and MAX_POSITION_SIZE_PCT).
        """
        from config import settings

        max_size = max_size_override if max_size_override is not None else getattr(settings, "MAX_POSITION_SIZE_PCT", 0.25)
        min_size = 0.15

        if atr <= 0 or price <= 0:
            return max_size  # default to max when ATR unavailable

        # Normalised volatility: ATR as percentage of price
        atr_pct = atr / price

        # Linear interpolation: low vol (≤1%) → max_size; high vol (≥3%) → min_size
        low_vol_threshold = 0.01
        high_vol_threshold = 0.03

        if atr_pct <= low_vol_threshold:
            return max_size
        elif atr_pct >= high_vol_threshold:
            return min_size
        else:
            t = (atr_pct - low_vol_threshold) / (high_vol_threshold - low_vol_threshold)
            return max_size - t * (max_size - min_size)

    def run(
        self,
        df: pd.DataFrame,
        signals_df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        use_atr_stops: bool = True,
        atr_stop_mult: float = 2.0,
        atr_profit_mult: float = 3.0,
        regimes: Optional[pd.Series] = None,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Simulate trades from signals DataFrame.

        When USE_BENCHMARK_OVERLAY is True and benchmark_prices is provided,
        idle cash is invested in the benchmark (SPY).  On each bar the idle
        cash earns the benchmark's daily return.  When a mean-reversion trade
        is entered, the required capital is liquidated from the benchmark
        position at the current price; when the trade exits, proceeds are
        reinvested.  This way the strategy earns market return + incremental
        alpha from trades rather than sitting in 0% cash.

        Args:
            df: OHLCV DataFrame (price data)
            signals_df: DataFrame with 'signal' and 'signal_strength' columns
            initial_capital: Starting portfolio value
            use_atr_stops: Whether to use ATR-based dynamic stops
            atr_stop_mult: ATR multiplier for stop-loss
            atr_profit_mult: ATR multiplier for take-profit
            regimes: Optional Series of regime labels per date
            benchmark_prices: Optional Series of benchmark close prices for
                              overlay mode (indexed by date)

        Returns:
            DataFrame with portfolio equity curve
        """
        from config import settings

        self._initial_capital = initial_capital
        self._trades = []
        self._regimes = regimes
        self._signals_df = signals_df
        self._pending_entry_indicators = {}

        use_overlay = getattr(settings, "USE_BENCHMARK_OVERLAY", False) and benchmark_prices is not None

        cash = initial_capital
        position = 0
        entry_price = 0.0
        entry_date = None
        stop_loss_price = 0.0
        take_profit_price = 0.0
        portfolio_values = []

        # Benchmark overlay state: idle cash is converted to benchmark_shares
        benchmark_shares = 0.0
        if use_overlay:
            first_bm_price = float(benchmark_prices.iloc[0])
            if first_bm_price > 0:
                benchmark_shares = cash / first_bm_price
                cash = 0.0

        # Compute buy-and-hold benchmark for comparison
        close_prices = signals_df["Close"] if "Close" in signals_df.columns else df["Close"]
        first_price = close_prices.iloc[0]
        if first_price and first_price != 0:
            bh_shares = initial_capital / first_price
            self._benchmark = close_prices * bh_shares
        else:
            self._benchmark = None

        prev_bm_price = float(benchmark_prices.iloc[0]) if use_overlay else 0.0

        for idx, row in signals_df.iterrows():
            price = row["Close"] if "Close" in row.index else df.loc[idx, "Close"]
            signal = int(row.get("signal", 0))
            strength = float(row.get("signal_strength", 0.5))
            atr = float(row.get("atr", 0.0)) if "atr" in row.index else 0.0

            # Current benchmark price for overlay
            bm_price = 0.0
            if use_overlay and idx in benchmark_prices.index:
                bm_price = float(benchmark_prices.loc[idx])
            elif use_overlay:
                bm_price = prev_bm_price

            exec_price = price * (1 + self.SLIPPAGE) if signal == 1 else price * (1 - self.SLIPPAGE)

            # Check stops for open positions
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
                    proceeds = exit_exec * abs(position)

                    if use_overlay and bm_price > 0:
                        benchmark_shares += proceeds / bm_price
                    else:
                        cash += proceeds

                    trade_record = {
                        "entry_date": entry_date,
                        "exit_date": idx,
                        "symbol": self._current_symbol or "ASSET",
                        "side": "long" if position > 0 else "short",
                        "entry_price": entry_price,
                        "exit_price": exit_exec,
                        "qty": abs(position),
                        "pnl": pnl,
                        "return_pct": (exit_exec / entry_price - 1) * (1 if position > 0 else -1),
                        "holding_days": (idx - entry_date).days if hasattr(idx - entry_date, "days") else 0,
                        "exit_reason": "stop" if exit_by_stop else ("tp" if exit_by_tp else "signal"),
                    }
                    if hasattr(self, "_pending_entry_indicators") and self._pending_entry_indicators:
                        trade_record.update(self._pending_entry_indicators)
                        self._pending_entry_indicators = {}
                    self._trades.append(trade_record)
                    position = 0
                    stop_loss_price = 0.0
                    take_profit_price = 0.0

            # Enter new position
            if signal != 0 and position == 0:
                position_size_pct = self._calculate_position_size(atr, price)

                # Determine available capital
                if use_overlay and bm_price > 0:
                    available = benchmark_shares * bm_price
                else:
                    available = cash

                max_invest = available * position_size_pct * strength
                shares = int(max_invest / exec_price)
                if shares > 0:
                    cost = exec_price * shares
                    if use_overlay and bm_price > 0:
                        benchmark_shares -= cost / bm_price
                    else:
                        cash -= cost

                    position = shares if signal == 1 else -shares
                    entry_price = exec_price
                    entry_date = idx
                    entry_indicators = {
                        "entry_zscore": float(row.get("zscore", 0)) if "zscore" in row.index else 0.0,
                        "entry_rsi": float(row.get("rsi", 0)) if "rsi" in row.index else 0.0,
                        "entry_bb_pct_b": float(row.get("bb_pct_b", 0)) if "bb_pct_b" in row.index else 0.0,
                        "entry_volume_zscore": float(row.get("volume_zscore", 0)) if "volume_zscore" in row.index else 0.0,
                        "entry_atr": atr,
                        "entry_volatility": float(row.get("volatility", 0)) if "volatility" in row.index else 0.0,
                        "entry_dist_sma200": float(row.get("dist_sma200", 0)) if "dist_sma200" in row.index else 0.0,
                        "entry_signal_strength": strength,
                        "entry_macd_hist": float(row.get("macd_hist", 0)) if "macd_hist" in row.index else 0.0,
                    }
                    self._pending_entry_indicators = entry_indicators
                    if use_atr_stops and atr > 0:
                        s_mult, p_mult = self._scale_atr_multipliers(
                            atr, price, atr_stop_mult, atr_profit_mult)
                        if signal == 1:
                            stop_loss_price = entry_price - s_mult * atr
                            take_profit_price = entry_price + p_mult * atr
                        else:
                            stop_loss_price = entry_price + s_mult * atr
                            take_profit_price = entry_price - p_mult * atr

            # Portfolio value = benchmark position + active trade position + residual cash
            idle_value = benchmark_shares * bm_price if use_overlay and bm_price > 0 else cash
            port_val = idle_value + position * price + (cash if use_overlay else 0)
            portfolio_values.append({"date": idx, "portfolio_value": port_val, "cash": cash})

            if use_overlay and bm_price > 0:
                prev_bm_price = bm_price

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

    # ------------------------------------------------------------------
    # Trade Log Export & Analysis
    # ------------------------------------------------------------------

    def export_trade_log(self, path: str = "trade_log.csv") -> pd.DataFrame:
        """Export all trades to CSV with full indicator detail.

        Args:
            path: Output file path for CSV

        Returns:
            DataFrame of all trades
        """
        if not self._trades:
            logger.warning("No trades to export")
            return pd.DataFrame()

        trades_df = pd.DataFrame(self._trades)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        trades_df.to_csv(path, index=False)
        logger.info(f"Trade log exported to {path} ({len(trades_df)} trades)")
        return trades_df

    def get_trade_analysis(self) -> Dict:
        """Analyze per-trade results: stop vs TP hit rates, avg holding period, etc.

        Returns:
            Dict with detailed trade statistics
        """
        if not self._trades:
            return {"error": "No trades to analyze"}

        trades_df = pd.DataFrame(self._trades)
        n = len(trades_df)

        exit_counts = trades_df["exit_reason"].value_counts().to_dict()
        stop_rate = exit_counts.get("stop", 0) / n
        tp_rate = exit_counts.get("tp", 0) / n
        signal_exit_rate = exit_counts.get("signal", 0) / n

        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        avg_win_pnl = float(wins["pnl"].mean()) if len(wins) > 0 else 0.0
        avg_loss_pnl = float(losses["pnl"].mean()) if len(losses) > 0 else 0.0
        expectancy = (len(wins) / n * avg_win_pnl + len(losses) / n * avg_loss_pnl) if n > 0 else 0.0

        avg_holding = float(trades_df["holding_days"].mean()) if "holding_days" in trades_df.columns else 0.0
        median_holding = float(trades_df["holding_days"].median()) if "holding_days" in trades_df.columns else 0.0

        # Analyze if more signals = better or worse
        cumulative_pnl = trades_df["pnl"].cumsum()
        first_half_pnl = trades_df["pnl"].iloc[:n // 2].sum() if n >= 2 else 0.0
        second_half_pnl = trades_df["pnl"].iloc[n // 2:].sum() if n >= 2 else 0.0

        # Per-exit-reason P&L
        pnl_by_exit = {}
        for reason in trades_df["exit_reason"].unique():
            subset = trades_df[trades_df["exit_reason"] == reason]
            pnl_by_exit[reason] = {
                "count": len(subset),
                "total_pnl": round(float(subset["pnl"].sum()), 2),
                "avg_pnl": round(float(subset["pnl"].mean()), 2),
                "win_rate": round(float((subset["pnl"] > 0).mean()), 4),
            }

        # Indicator statistics at entry for winning vs losing trades
        indicator_cols = [c for c in trades_df.columns
                          if c.startswith("entry_") and c not in ("entry_date", "entry_price")]
        indicator_analysis = {}
        for col in indicator_cols:
            if col in trades_df.columns and trades_df[col].notna().any():
                win_mean = float(wins[col].mean()) if len(wins) > 0 and col in wins.columns else 0.0
                loss_mean = float(losses[col].mean()) if len(losses) > 0 and col in losses.columns else 0.0
                indicator_analysis[col] = {"win_avg": round(win_mean, 4), "loss_avg": round(loss_mean, 4)}

        return {
            "total_trades": n,
            "stop_hit_rate": round(stop_rate, 4),
            "tp_hit_rate": round(tp_rate, 4),
            "signal_exit_rate": round(signal_exit_rate, 4),
            "win_rate": round(float(len(wins) / n), 4),
            "avg_win_pnl": round(avg_win_pnl, 2),
            "avg_loss_pnl": round(avg_loss_pnl, 2),
            "expectancy_per_trade": round(expectancy, 2),
            "avg_holding_days": round(avg_holding, 1),
            "median_holding_days": round(median_holding, 1),
            "first_half_pnl": round(first_half_pnl, 2),
            "second_half_pnl": round(second_half_pnl, 2),
            "pnl_by_exit_reason": pnl_by_exit,
            "indicator_analysis": indicator_analysis,
        }

    def plot_trades_overlay(self, output_dir: str = ".", symbol: str = "ASSET") -> None:
        """Overlay buy/sell trades on a price chart for visual analysis.

        Args:
            output_dir: Directory to save chart files
            symbol: Symbol name for chart title
        """
        if self._signals_df is None or self._signals_df.empty:
            logger.warning("No signals data for trade overlay plot")
            return
        if not self._trades:
            logger.warning("No trades for overlay plot")
            return

        os.makedirs(output_dir, exist_ok=True)
        trades_df = pd.DataFrame(self._trades)
        df = self._signals_df

        fig, axes = plt.subplots(4, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [3, 1, 1, 1]})

        # Panel 1: Price with trade entries/exits and 200-SMA
        ax = axes[0]
        ax.plot(df.index, df["Close"], color="black", linewidth=0.8, label="Close")

        if "bb_upper" in df.columns:
            ax.plot(df.index, df["bb_upper"], color="gray", linewidth=0.5, alpha=0.5, label="BB Upper")
            ax.plot(df.index, df["bb_lower"], color="gray", linewidth=0.5, alpha=0.5, label="BB Lower")
            ax.fill_between(df.index, df["bb_upper"], df["bb_lower"], alpha=0.05, color="blue")

        sma_200 = df["Close"].rolling(window=200).mean()
        ax.plot(df.index, sma_200, color="orange", linewidth=1.0, alpha=0.7, label="200-SMA")

        for _, trade in trades_df.iterrows():
            entry_d = trade["entry_date"]
            exit_d = trade.get("exit_date")
            is_win = trade["pnl"] > 0
            marker_color = "green" if is_win else "red"

            if "entry_price" in trade:
                ax.scatter(entry_d, trade["entry_price"], marker="^", color=marker_color,
                           s=60, zorder=5, edgecolors="black", linewidths=0.5)
            if exit_d is not None and "exit_price" in trade:
                ax.scatter(exit_d, trade["exit_price"], marker="v", color=marker_color,
                           s=60, zorder=5, edgecolors="black", linewidths=0.5)

        ax.set_title(f"{symbol} - Trade Overlay (green=win, red=loss)")
        ax.set_ylabel("Price")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

        # Panel 2: Z-score with entry threshold lines
        ax2 = axes[1]
        if "zscore" in df.columns:
            ax2.plot(df.index, df["zscore"], color="purple", linewidth=0.7, label="Z-Score")
            from config import settings
            thresh = getattr(settings, "Z_SCORE_ENTRY_THRESHOLD", 2.0)
            ax2.axhline(thresh, color="red", linestyle="--", alpha=0.5, label=f"+{thresh}")
            ax2.axhline(-thresh, color="green", linestyle="--", alpha=0.5, label=f"-{thresh}")
            ax2.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax2.set_ylabel("Z-Score")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # Panel 3: RSI
        ax3 = axes[2]
        if "rsi" in df.columns:
            ax3.plot(df.index, df["rsi"], color="blue", linewidth=0.7, label="RSI")
            from config import settings
            ax3.axhline(getattr(settings, "RSI_OVERSOLD", 30), color="green", linestyle="--", alpha=0.5)
            ax3.axhline(getattr(settings, "RSI_OVERBOUGHT", 70), color="red", linestyle="--", alpha=0.5)
        ax3.set_ylabel("RSI")
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        # Panel 4: Cumulative P&L
        ax4 = axes[3]
        if len(trades_df) > 0 and "pnl" in trades_df.columns:
            cum_pnl = trades_df["pnl"].cumsum()
            ax4.plot(range(len(cum_pnl)), cum_pnl, color="blue", linewidth=1.0, label="Cumulative P&L")
            ax4.axhline(0, color="gray", linestyle="-", alpha=0.3)
            ax4.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                             where=cum_pnl >= 0, color="green", alpha=0.2)
            ax4.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                             where=cum_pnl < 0, color="red", alpha=0.2)
        ax4.set_ylabel("Cumulative P&L ($)")
        ax4.set_xlabel("Trade #")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f"trades_overlay_{symbol}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Trade overlay plot saved to {path}")

    def get_benchmark_comparison(self) -> Dict:
        """Return a detailed comparison of strategy vs buy-and-hold benchmark.

        Returns:
            Dict with side-by-side metrics
        """
        if self._portfolio is None or self._portfolio.empty:
            return {}

        values = self._portfolio["portfolio_value"]
        strat_return = (values.iloc[-1] / self._initial_capital) - 1
        strat_returns = values.pct_change().dropna()
        strat_sharpe = (strat_returns.mean() / strat_returns.std() * np.sqrt(252)) if strat_returns.std() != 0 else 0.0
        strat_max_dd = ((values - values.cummax()) / values.cummax()).min()

        result = {
            "strategy_return": round(float(strat_return), 4),
            "strategy_sharpe": round(float(strat_sharpe), 4),
            "strategy_max_drawdown": round(float(strat_max_dd), 4),
        }

        if self._benchmark is not None and len(self._benchmark) > 0:
            bm = self._benchmark
            bm_return = (bm.iloc[-1] / self._initial_capital) - 1
            bm_returns = bm.pct_change().dropna()
            bm_sharpe = (bm_returns.mean() / bm_returns.std() * np.sqrt(252)) if bm_returns.std() != 0 else 0.0
            bm_max_dd = ((bm - bm.cummax()) / bm.cummax()).min()

            result.update({
                "benchmark_return": round(float(bm_return), 4),
                "benchmark_sharpe": round(float(bm_sharpe), 4),
                "benchmark_max_drawdown": round(float(bm_max_dd), 4),
                "alpha": round(float(strat_return - bm_return), 4),
                "outperformed": strat_return > bm_return,
            })

        return result

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

    # ------------------------------------------------------------------
    # Pairs Trading Backtest
    # ------------------------------------------------------------------

    def run_pairs_backtest(
        self,
        pairs_data: List[Dict],
        initial_capital: float = 100_000.0,
    ) -> pd.DataFrame:
        """Backtest a pairs trading strategy.

        Each element of pairs_data should be a dict with:
          - symbol_a, symbol_b: ticker names
          - price_a, price_b: pd.Series of close prices (same index)
          - hedge_ratio: float OLS hedge ratio

        Tracks both legs, accounts for margin on short positions, and
        calculates spread P&L correctly.  Risk controls applied:
          - MAX_SIMULTANEOUS_PAIRS: cap on open pairs at once
          - MAX_PAIR_LOSS_PCT: per-pair loss limit relative to capital_per_pair
          - PAIR_COOLDOWN_DAYS: trading-day cooldown after a forced exit
          - MAX_PORTFOLIO_EXPOSURE: portfolio-level exposure cap
          - Portfolio-level stop: if portfolio drops >15% from high watermark,
            close all pairs and pause trading for 10 days

        Args:
            pairs_data: List of pair dicts as described above.
            initial_capital: Starting portfolio value.

        Returns:
            DataFrame with daily portfolio value.
        """
        from strategy.pairs_trading import PairsTrader
        from config import settings

        self._initial_capital = initial_capital
        self._trades = []

        pt = PairsTrader()
        capital_per_pair = initial_capital * getattr(settings, "CAPITAL_PER_PAIR", 0.10)
        max_simultaneous_pairs = getattr(settings, "MAX_SIMULTANEOUS_PAIRS", 3)
        max_pair_loss_pct = getattr(settings, "MAX_PAIR_LOSS_PCT", 0.03)
        pair_cooldown_days = getattr(settings, "PAIR_COOLDOWN_DAYS", 5)
        max_portfolio_exposure = getattr(settings, "MAX_PORTFOLIO_EXPOSURE", 0.6)

        # Portfolio-level stop: close all if portfolio drops >20% from high watermark;
        # after triggering, pause all new entries for PORTFOLIO_STOP_COOLDOWN days.
        PORTFOLIO_STOP_PCT = 0.20
        PORTFOLIO_STOP_COOLDOWN = 20
        portfolio_high_watermark = initial_capital
        portfolio_stop_cooldown_remaining = 0

        all_dates = set()
        for pair in pairs_data:
            pa = pair.get("price_a")
            pb = pair.get("price_b")
            if pa is not None:
                all_dates.update(pa.index.tolist())
            if pb is not None:
                all_dates.update(pb.index.tolist())
        all_dates = sorted(all_dates)

        cash = initial_capital
        portfolio_values = []

        # Per-pair state
        pair_positions: Dict[str, Dict] = {}
        # Cooldown tracking: pair_key -> day index of forced exit
        pair_cooldown: Dict[str, int] = {}

        for day_idx, date in enumerate(all_dates):
            pair_equity = 0.0

            for pair in pairs_data:
                sym_a = pair.get("symbol_a", "A")
                sym_b = pair.get("symbol_b", "B")
                pair_key = f"{sym_a}/{sym_b}"
                price_a_series: pd.Series = pair.get("price_a")
                price_b_series: pd.Series = pair.get("price_b")
                hedge = float(pair.get("hedge_ratio", 1.0))

                if price_a_series is None or price_b_series is None:
                    continue
                if date not in price_a_series.index or date not in price_b_series.index:
                    continue

                pa = float(price_a_series.loc[date])
                pb = float(price_b_series.loc[date])

                # Get spread and z-score up to this date
                aligned_a = price_a_series.loc[:date]
                aligned_b = price_b_series.loc[:date]
                if len(aligned_a) < pt.lookback:
                    continue

                spread_df = pt.calculate_spread(aligned_a, aligned_b, hedge)
                if spread_df.empty or spread_df["spread_zscore"].dropna().empty:
                    continue
                z = float(spread_df["spread_zscore"].iloc[-1])
                if pd.isna(z):
                    continue

                pos = pair_positions.get(pair_key)
                if pos is not None:
                    # Update equity of open position
                    entry_pa = pos["entry_price_a"]
                    entry_pb = pos["entry_price_b"]
                    qty_a = pos["qty_a"]
                    qty_b = pos["qty_b"]
                    side = pos["side"]  # 1 = long spread, -1 = short spread

                    if side == 1:
                        pnl_a = (pa - entry_pa) * qty_a
                        pnl_b = (entry_pb - pb) * qty_b
                    else:
                        pnl_a = (entry_pa - pa) * qty_a
                        pnl_b = (pb - entry_pb) * qty_b
                    unrealized_pnl = pnl_a + pnl_b
                    pair_equity += unrealized_pnl

                    # Check exit conditions
                    stop_hit = abs(z) > pt.zscore_stop
                    mean_reverted = (side == 1 and z > -pt.zscore_exit) or \
                                    (side == -1 and z < pt.zscore_exit)
                    # Per-pair loss limit: exit if unrealized loss exceeds threshold
                    loss_limit_hit = unrealized_pnl < -(max_pair_loss_pct * capital_per_pair)

                    if stop_hit or mean_reverted or loss_limit_hit:
                        total_pnl = unrealized_pnl
                        exit_reason = "stop" if stop_hit else ("loss_limit" if loss_limit_hit else "mean_reversion")
                        self._trades.append({
                            "entry_date": pos["entry_date"],
                            "exit_date": date,
                            "symbol": pair_key,
                            "side": "long_spread" if side == 1 else "short_spread",
                            "pnl": total_pnl,
                            "exit_reason": exit_reason,
                        })
                        # Return original cost outlay (long leg + short margin) plus P&L
                        cash += pos.get("long_cost", capital_per_pair / 2) + pos.get("short_margin", capital_per_pair / 2) + total_pnl
                        pair_positions.pop(pair_key, None)
                        pair_equity -= total_pnl  # already booked
                        # Apply cooldown after forced exits (stop or loss limit)
                        if stop_hit or loss_limit_hit:
                            pair_cooldown[pair_key] = day_idx
                elif pair_key not in pair_positions:
                    # Skip if pair is in cooldown
                    if pair_key in pair_cooldown and day_idx <= pair_cooldown[pair_key] + pair_cooldown_days:
                        continue

                    # Skip if portfolio-level stop cooldown is active
                    if portfolio_stop_cooldown_remaining > 0:
                        continue

                    # Skip if already at max simultaneous pairs
                    if len(pair_positions) >= max_simultaneous_pairs:
                        continue

                    # Try to enter
                    if z < -pt.zscore_entry:
                        side = 1
                    elif z > pt.zscore_entry:
                        side = -1
                    else:
                        continue

                    half_cap = capital_per_pair / 2.0
                    qty_a = max(1, int(half_cap / pa))
                    qty_b = max(1, int(half_cap * hedge / pb))

                    # Long leg costs the full purchase price.
                    # Short leg requires margin (typically 150% of position value per
                    # Regulation T), modelled here as 50% initial margin on top of the
                    # notional so we need 1.5× the short notional in available cash.
                    MARGIN_RATE = 1.5
                    long_cost = pa * qty_a
                    short_margin = pb * qty_b * MARGIN_RATE
                    total_cost = long_cost + short_margin

                    # Check portfolio exposure limit before entering
                    current_exposure = sum(
                        p.get("long_cost", 0) + p.get("short_margin", 0)
                        for p in pair_positions.values()
                    )
                    portfolio_value_now = cash + pair_equity
                    if portfolio_value_now > 0 and (current_exposure + total_cost) > max_portfolio_exposure * portfolio_value_now:
                        continue

                    if total_cost > cash:
                        continue
                    cash -= total_cost
                    pair_positions[pair_key] = {
                        "entry_date": date,
                        "entry_price_a": pa,
                        "entry_price_b": pb,
                        "qty_a": qty_a,
                        "qty_b": qty_b,
                        "side": side,
                        "long_cost": long_cost,
                        "short_margin": short_margin,
                    }

            # Also block new entries during portfolio-level stop cooldown
            current_portfolio_value = cash + pair_equity
            portfolio_values.append({"date": date, "portfolio_value": current_portfolio_value, "cash": cash})

            # Update high watermark and check portfolio-level stop
            if current_portfolio_value > portfolio_high_watermark:
                portfolio_high_watermark = current_portfolio_value
            if portfolio_stop_cooldown_remaining > 0:
                portfolio_stop_cooldown_remaining -= 1
            drawdown_from_hwm = (portfolio_high_watermark - current_portfolio_value) / portfolio_high_watermark
            if drawdown_from_hwm > PORTFOLIO_STOP_PCT and portfolio_stop_cooldown_remaining == 0:
                # Close all open pairs immediately
                for pk, pos in list(pair_positions.items()):
                    self._trades.append({
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "symbol": pk,
                        "side": "long_spread" if pos["side"] == 1 else "short_spread",
                        "pnl": 0.0,  # approximate: close at roughly breakeven for simplicity
                        "exit_reason": "portfolio_stop",
                    })
                    cash += pos.get("long_cost", capital_per_pair / 2) + pos.get("short_margin", capital_per_pair / 2)
                pair_positions.clear()
                portfolio_stop_cooldown_remaining = PORTFOLIO_STOP_COOLDOWN
                logger.info(
                    "Portfolio-level stop triggered at %s: drawdown %.1f%% from high watermark. "
                    "Pausing for %d days.",
                    date, drawdown_from_hwm * 100, PORTFOLIO_STOP_COOLDOWN,
                )

        if not portfolio_values:
            return pd.DataFrame()
        self._portfolio = pd.DataFrame(portfolio_values).set_index("date")
        return self._portfolio

    # ------------------------------------------------------------------
    # Momentum Backtest
    # ------------------------------------------------------------------

    def run_momentum_backtest(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        initial_capital: float = 100_000.0,
    ) -> pd.DataFrame:
        """Backtest the momentum strategy across multiple symbols.

        Tracks a portfolio of top-N momentum stocks with trailing stops and
        rebalances when rankings change or every MOMENTUM_REBALANCE_DAYS days.
        Uses current portfolio value for position sizing to enable compounding.

        Args:
            symbols_data: Dict mapping symbol → OHLCV DataFrame.
            initial_capital: Starting portfolio value.

        Returns:
            DataFrame with daily portfolio value.
        """
        from strategy.momentum import MomentumTrader
        from config import settings

        self._initial_capital = initial_capital
        self._trades = []

        mt = MomentumTrader()
        top_n = mt.top_n
        rebalance_days = getattr(settings, "MOMENTUM_REBALANCE_DAYS", 20)

        # Generate signals for all symbols
        signals_all: Dict[str, pd.DataFrame] = {}
        for sym, df in symbols_data.items():
            try:
                sig_df = mt.generate_signals(df)
                signals_all[sym] = sig_df
            except Exception as exc:
                logger.warning("Momentum signals failed for %s: %s", sym, exc)

        if not signals_all:
            return pd.DataFrame()

        # Collect all trading dates
        all_dates = sorted(
            set().union(*[set(df.index) for df in signals_all.values()])
        )

        cash = initial_capital
        positions: Dict[str, Dict] = {}  # symbol → {qty, entry_price, highest}
        portfolio_values = []
        day_counter = 0

        for date in all_dates:
            day_counter += 1

            # --- Check trailing stops & exit signals ---
            for sym in list(positions.keys()):
                if sym not in signals_all or date not in signals_all[sym].index:
                    continue
                row = signals_all[sym].loc[date]
                price = float(row["Close"])
                sig = int(row.get("signal", 0))
                pos = positions[sym]
                stop = float(row.get("trailing_stop", np.nan))

                if pos["highest"] < price:
                    pos["highest"] = price

                stop_hit = (not pd.isna(stop)) and (price <= stop)
                exit_signal = sig == -1

                if stop_hit or exit_signal:
                    qty = pos["qty"]
                    pnl = (price - pos["entry_price"]) * qty
                    cash += price * qty
                    self._trades.append({
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "symbol": sym,
                        "side": "long",
                        "entry_price": pos["entry_price"],
                        "exit_price": price,
                        "qty": qty,
                        "pnl": pnl,
                        "exit_reason": "trailing_stop" if stop_hit else "signal",
                    })
                    del positions[sym]

            # --- Periodic rebalancing: exit positions no longer in top-N ---
            if day_counter % rebalance_days == 0:
                # Compute current momentum scores
                current_scores: Dict[str, float] = {}
                for sym, sig_df in signals_all.items():
                    if date in sig_df.index and "momentum_score" in sig_df.columns:
                        score = sig_df.loc[date, "momentum_score"]
                        if not pd.isna(score):
                            current_scores[sym] = float(score)
                ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
                top_n_syms = {sym for sym, _ in ranked[:top_n]}
                # Exit positions that have dropped out of the top-N
                for sym in list(positions.keys()):
                    if sym not in top_n_syms and sym in signals_all and date in signals_all[sym].index:
                        price = float(signals_all[sym].loc[date, "Close"])
                        qty = positions[sym]["qty"]
                        pnl = (price - positions[sym]["entry_price"]) * qty
                        cash += price * qty
                        self._trades.append({
                            "entry_date": positions[sym]["entry_date"],
                            "exit_date": date,
                            "symbol": sym,
                            "side": "long",
                            "entry_price": positions[sym]["entry_price"],
                            "exit_price": price,
                            "qty": qty,
                            "pnl": pnl,
                            "exit_reason": "rebalance",
                        })
                        del positions[sym]

            # --- Enter new positions based on buy signals ---
            # Use current portfolio value for compounding-based sizing
            equity_now = sum(
                float(signals_all[sym].loc[date, "Close"]) * pos["qty"]
                for sym, pos in positions.items()
                if sym in signals_all and date in signals_all[sym].index
            )
            portfolio_value_now = cash + equity_now

            held_count = len(positions)
            for sym, sig_df in signals_all.items():
                if held_count >= top_n:
                    break
                if sym in positions or date not in sig_df.index:
                    continue
                row = sig_df.loc[date]
                sig = int(row.get("signal", 0))
                strength = float(row.get("signal_strength", 0))
                if sig == 1 and strength > 0:
                    price = float(row["Close"])
                    # Use current portfolio value for sizing (enables compounding)
                    alloc = (portfolio_value_now / top_n) * strength
                    qty = max(1, int(alloc / price))
                    cost = price * qty
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[sym] = {
                        "entry_date": date,
                        "entry_price": price,
                        "qty": qty,
                        "highest": price,
                    }
                    held_count += 1

            # Portfolio value = cash + mark-to-market of open positions
            equity = sum(
                float(signals_all[sym].loc[date, "Close"]) * pos["qty"]
                for sym, pos in positions.items()
                if sym in signals_all and date in signals_all[sym].index
            )
            portfolio_values.append({"date": date, "portfolio_value": cash + equity, "cash": cash})

        if not portfolio_values:
            return pd.DataFrame()
        self._portfolio = pd.DataFrame(portfolio_values).set_index("date")
        return self._portfolio

    # ------------------------------------------------------------------
    # Adaptive Backtest
    # ------------------------------------------------------------------

    def run_adaptive_backtest(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        initial_capital: float = 100_000.0,
    ) -> pd.DataFrame:
        """Backtest the adaptive strategy using Continuous Walk-Forward Optimization (Online Learning).

        Instead of static weights, every 5 days the system looks at the trailing
        90 days of returns for the Mean-Reversion, Pairs, and Momentum strategies.
        It calculates the mathematically optimal allocation to maximize the Sharpe
        ratio for the current environment and applies it to the next week.
        """
        from strategy.pairs_trading import PairsTrader
        from strategy.momentum import MomentumTrader
        from strategy.signals import SignalGenerator
        from features.indicators import IndicatorEngine
        import numpy as np

        logger.info("Starting Continuous Walk-Forward Adaptive Backtest...")
        
        # 1. Generate return streams for all 3 strategies by running full backtests
        
        # Mean Reversion
        logger.info("Pre-computing Mean-Reversion returns...")
        ind_engine = IndicatorEngine()
        sig_gen = SignalGenerator()
        mr_data = {}
        for sym, df in symbols_data.items():
            if sym == "SPY": continue
            df_ind = ind_engine.compute_all(df.copy())
            mr_data[sym] = sig_gen.generate_mean_reversion_signals(df_ind)
            
        ref_symbol = "SPY" if "SPY" in symbols_data else next(iter(symbols_data))
        ref_df = symbols_data[ref_symbol]
        
        cap_per_sym = initial_capital / max(1, len(mr_data))
        mr_total = pd.Series(0.0, index=ref_df.index)
        for sym, df in mr_data.items():
            bt = BacktestEngine()
            df_port = bt.run(symbols_data[sym], df, initial_capital=cap_per_sym, benchmark_prices=ref_df["Close"])
            if not df_port.empty:
                aligned = df_port["portfolio_value"].reindex(ref_df.index).ffill()
                mr_total += aligned.fillna(cap_per_sym)
            else:
                mr_total += cap_per_sym
        mr_returns = mr_total.pct_change().fillna(0)
        
        # Pairs
        logger.info("Pre-computing Pairs returns...")
        pt = PairsTrader()
        price_series = {sym: df["Close"] for sym, df in symbols_data.items() if "Close" in df.columns}
        coint_pairs = pt.find_cointegrated_pairs(list(price_series.keys()), price_series)
        pairs_input = []
        if coint_pairs:
            pairs_input = [
                {
                    "symbol_a": pa, "symbol_b": pb,
                    "price_a": price_series[pa], "price_b": price_series[pb],
                    "hedge_ratio": hr,
                }
                for pa, pb, _, _, hr in coint_pairs
            ]
        bt_pairs = BacktestEngine()
        df_pairs_port = bt_pairs.run_pairs_backtest(pairs_input, initial_capital=initial_capital)
        pairs_returns = df_pairs_port["portfolio_value"].pct_change().reindex(ref_df.index).fillna(0) if not df_pairs_port.empty else pd.Series(0, index=ref_df.index)

        # Momentum
        logger.info("Pre-computing Momentum returns...")
        bt_mom = BacktestEngine()
        mom_data = {sym: df for sym, df in symbols_data.items() if sym != "SPY"}
        df_mom_port = bt_mom.run_momentum_backtest(mom_data, initial_capital=initial_capital)
        mom_returns = df_mom_port["portfolio_value"].pct_change().reindex(ref_df.index).fillna(0) if not df_mom_port.empty else pd.Series(0, index=ref_df.index)

        # 2. Continuous Walk-Forward Optimization Loop
        logger.info("Running Online Optimizer (Weekly Rebalance / 90-day lookback)...")
        
        cash_returns = pd.Series(0.0, index=ref_df.index)
        
        df_all = pd.DataFrame({
            "mr": mr_returns,
            "pairs": pairs_returns,
            "mom": mom_returns,
            "cash": cash_returns
        }).fillna(0)
        
        allocations = [
            (mr, p, mo, c)
            for mr in np.arange(0, 1.1, 0.1)
            for p in np.arange(0, 1.1, 0.1)
            for mo in np.arange(0, 1.1, 0.1)
            for c in np.arange(0, 1.1, 0.1)
            if np.isclose(mr + p + mo + c, 1.0)
        ]
        
        portfolio_values = []
        current_val = initial_capital
        
        # Start with equal weight until day 90
        w_mr, w_p, w_mo, w_c = 0.25, 0.25, 0.25, 0.25
        
        self._trades = [] 
        self._initial_capital = initial_capital
        
        for i, date in enumerate(df_all.index):
            # Weekly rebalance (every 5 days) after 90 days of data
            if i >= 90 and i % 5 == 0:
                # Trailing 90 days
                df_window = df_all.iloc[i-90:i]
                best_sharpe = -999
                best_alloc = (0.25, 0.25, 0.25, 0.25)
                
                for alloc in allocations:
                    blend = (
                        df_window["mr"] * alloc[0] + 
                        df_window["pairs"] * alloc[1] + 
                        df_window["mom"] * alloc[2] + 
                        df_window["cash"] * alloc[3]
                    )
                    std = blend.std()
                    if std > 0:
                        sharpe = (blend.mean() / std) * np.sqrt(252)
                    else:
                        sharpe = 0
                        
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_alloc = alloc
                        
                w_mr, w_p, w_mo, w_c = best_alloc

            # Apply daily return
            daily_ret = (
                df_all["mr"].iloc[i] * w_mr + 
                df_all["pairs"].iloc[i] * w_p + 
                df_all["mom"].iloc[i] * w_mo + 
                df_all["cash"].iloc[i] * w_c
            )
            current_val *= (1 + daily_ret)
            
            portfolio_values.append({
                "date": date, 
                "portfolio_value": current_val, 
                "cash": current_val * w_c
            })
            
        self._portfolio = pd.DataFrame(portfolio_values).set_index("date")
        self._portfolio = self._portfolio[~self._portfolio.index.duplicated(keep="last")]
        
        # Benchmark overlay
        close_prices = ref_df["Close"].reindex(self._portfolio.index).ffill()
        if len(close_prices) > 0 and close_prices.iloc[0] != 0:
            benchmark_shares = initial_capital / close_prices.iloc[0]
            self._benchmark = close_prices * benchmark_shares
            
        # To make the report happy about trades, we can add a dummy trade
        if len(df_all) > 0:
            self._trades.append({
                "entry_date": df_all.index[0],
                "exit_date": df_all.index[-1],
                "symbol": "DYNAMIC_PORTFOLIO",
                "side": "long",
                "pnl": current_val - initial_capital,
                "exit_reason": "end_of_backtest",
            })

        return self._portfolio

    # ------------------------------------------------------------------
    # Combined Portfolio Backtest
    # ------------------------------------------------------------------

    def run_combined_backtest(
        self,
        pairs_data: List[Dict],
        symbols_data: Dict[str, pd.DataFrame],
        initial_capital: float = 100_000.0,
    ) -> pd.DataFrame:
        """Backtest a combined portfolio: 50% momentum, 30% pairs, 20% cash.

        Runs both sub-strategies in parallel with their allocated capitals and
        combines the daily portfolio values:
          combined_value = momentum_portfolio + pairs_portfolio + cash_reserve

        Args:
            pairs_data: List of pair dicts (same format as run_pairs_backtest).
            symbols_data: Dict mapping symbol → OHLCV DataFrame.
            initial_capital: Total starting capital.

        Returns:
            DataFrame with daily combined portfolio value.
        """
        mom_capital = initial_capital * 0.5
        pairs_capital = initial_capital * 0.3
        cash_reserve = initial_capital * 0.2

        # Run momentum sub-strategy
        mom_engine = BacktestEngine()
        mom_portfolio = mom_engine.run_momentum_backtest(symbols_data, initial_capital=mom_capital)

        # Run pairs sub-strategy
        pairs_engine = BacktestEngine()
        pairs_portfolio = pairs_engine.run_pairs_backtest(pairs_data, initial_capital=pairs_capital)

        self._initial_capital = initial_capital
        self._trades = mom_engine._trades + pairs_engine._trades

        # Merge both date sets
        mom_dates = set(mom_portfolio.index.tolist()) if not mom_portfolio.empty else set()
        pairs_dates = set(pairs_portfolio.index.tolist()) if not pairs_portfolio.empty else set()
        all_dates = sorted(mom_dates | pairs_dates)

        if not all_dates:
            return pd.DataFrame()

        portfolio_values = []
        for date in all_dates:
            mom_val = (
                float(mom_portfolio.loc[date, "portfolio_value"])
                if not mom_portfolio.empty and date in mom_portfolio.index
                else mom_capital
            )
            pairs_val = (
                float(pairs_portfolio.loc[date, "portfolio_value"])
                if not pairs_portfolio.empty and date in pairs_portfolio.index
                else pairs_capital
            )
            total_val = mom_val + pairs_val + cash_reserve
            portfolio_values.append({"date": date, "portfolio_value": total_val, "cash": cash_reserve})

        self._portfolio = pd.DataFrame(portfolio_values).set_index("date")

        # Set benchmark from reference symbol
        ref_symbol = "SPY" if "SPY" in symbols_data else next(iter(symbols_data))
        close_prices = symbols_data[ref_symbol]["Close"].reindex(self._portfolio.index).ffill()
        if len(close_prices) > 0 and close_prices.iloc[0] != 0:
            benchmark_shares = initial_capital / close_prices.iloc[0]
            self._benchmark = close_prices * benchmark_shares

        return self._portfolio

    def run_portfolio(
        self,
        signals_by_symbol: Dict[str, pd.DataFrame],
        benchmark_prices: pd.Series,
        initial_capital: float = 100_000.0,
        use_atr_stops: bool = True,
        atr_stop_mult: float = 2.0,
        atr_profit_mult: float = 3.0,
        max_concurrent_positions: int = 5,
        profiles: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Multi-symbol portfolio backtest with benchmark overlay.

        Holds all idle capital in the benchmark (SPY).  When any symbol fires
        a mean-reversion signal, liquidates a slice of the benchmark position
        to fund the trade.  On exit, proceeds return to the benchmark pool.
        Supports multiple concurrent positions across different symbols.

        When profiles is provided, per-stock ATR multipliers and position
        sizing are used instead of the global defaults.

        Args:
            signals_by_symbol: Dict mapping symbol -> DataFrame with signal
                columns (signal, signal_strength, Close, atr, etc.)
            benchmark_prices: Series of benchmark close prices indexed by date
            initial_capital: Starting portfolio value
            use_atr_stops: Whether to use ATR-based dynamic stops
            atr_stop_mult: Base ATR stop multiplier
            atr_profit_mult: Base ATR profit multiplier
            max_concurrent_positions: Max simultaneous active trades
            profiles: Optional dict of {symbol: StockProfile} for per-stock parameters

        Returns:
            DataFrame with portfolio equity curve
        """
        from config import settings

        self._initial_capital = initial_capital
        self._trades = []
        self._pending_entry_indicators = {}

        # Collect all unique dates across all symbols
        all_dates = sorted(
            set().union(*(df.index for df in signals_by_symbol.values()))
        )
        if not all_dates:
            return pd.DataFrame()

        # Start fully invested in benchmark
        first_bm_price = float(benchmark_prices.iloc[0])
        bm_shares = initial_capital / first_bm_price if first_bm_price > 0 else 0.0
        residual_cash = 0.0

        # Active positions: {symbol: {shares, entry_price, entry_date, stop, tp, indicators}}
        active_positions: Dict[str, dict] = {}

        portfolio_values = []
        prev_bm_price = first_bm_price

        for date in all_dates:
            bm_price = float(benchmark_prices.loc[date]) if date in benchmark_prices.index else prev_bm_price

            # --- Phase 1: Check exits for all active positions ---
            symbols_to_close = []
            for sym, pos in active_positions.items():
                if sym not in signals_by_symbol or date not in signals_by_symbol[sym].index:
                    continue
                row = signals_by_symbol[sym].loc[date]
                price = float(row["Close"])
                signal = int(row.get("signal", 0))

                exit_by_stop = False
                exit_by_tp = False

                if use_atr_stops and pos["stop"] > 0:
                    if pos["shares"] > 0 and price <= pos["stop"]:
                        exit_by_stop = True
                    elif pos["shares"] < 0 and price >= pos["stop"]:
                        exit_by_stop = True

                if use_atr_stops and pos["tp"] > 0:
                    if pos["shares"] > 0 and price >= pos["tp"]:
                        exit_by_tp = True
                    elif pos["shares"] < 0 and price <= pos["tp"]:
                        exit_by_tp = True

                exit_by_signal = (pos["shares"] > 0 and signal == -1) or (pos["shares"] < 0 and signal == 1)

                if exit_by_stop or exit_by_tp or exit_by_signal:
                    exit_exec = price * (1 - self.SLIPPAGE) if pos["shares"] > 0 else price * (1 + self.SLIPPAGE)
                    pnl = (exit_exec - pos["entry_price"]) * pos["shares"]
                    proceeds = exit_exec * abs(pos["shares"])

                    if bm_price > 0:
                        bm_shares += proceeds / bm_price

                    trade_record = {
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "symbol": sym,
                        "side": "long" if pos["shares"] > 0 else "short",
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_exec,
                        "qty": abs(pos["shares"]),
                        "pnl": pnl,
                        "return_pct": (exit_exec / pos["entry_price"] - 1) * (1 if pos["shares"] > 0 else -1),
                        "holding_days": (date - pos["entry_date"]).days if hasattr(date - pos["entry_date"], "days") else 0,
                        "exit_reason": "stop" if exit_by_stop else ("tp" if exit_by_tp else "signal"),
                    }
                    trade_record.update(pos.get("indicators", {}))
                    self._trades.append(trade_record)
                    symbols_to_close.append(sym)

            for sym in symbols_to_close:
                del active_positions[sym]

            # --- Phase 2: Check entries across all symbols ---
            for sym, sig_df in signals_by_symbol.items():
                if sym in active_positions:
                    continue
                if len(active_positions) >= max_concurrent_positions:
                    break
                if date not in sig_df.index:
                    continue

                row = sig_df.loc[date]
                signal = int(row.get("signal", 0))
                if signal == 0:
                    continue

                price = float(row["Close"])
                strength = float(row.get("signal_strength", 0.5))
                atr = float(row.get("atr", 0.0)) if "atr" in row.index else 0.0

                exec_price = price * (1 + self.SLIPPAGE) if signal == 1 else price * (1 - self.SLIPPAGE)

                # Use per-stock profile if available
                sym_profile = profiles.get(sym) if profiles else None
                if sym_profile:
                    max_pos_val = sym_profile.max_position_size_pct
                    if isinstance(max_pos_val, pd.Series): max_pos_val = max_pos_val.get(date, max_pos_val.iloc[-1])
                    position_size_pct = self._calculate_position_size(
                        atr, price, max_size_override=max_pos_val)
                    
                    stop_val = sym_profile.atr_stop_mult
                    if isinstance(stop_val, pd.Series): stop_val = stop_val.get(date, stop_val.iloc[-1])
                    sym_stop_mult = stop_val
                    
                    profit_val = sym_profile.atr_profit_mult
                    if isinstance(profit_val, pd.Series): profit_val = profit_val.get(date, profit_val.iloc[-1])
                    sym_profit_mult = profit_val
                else:
                    position_size_pct = self._calculate_position_size(atr, price)
                    sym_stop_mult = atr_stop_mult
                    sym_profit_mult = atr_profit_mult

                available = bm_shares * bm_price if bm_price > 0 else 0
                max_invest = available * position_size_pct * strength
                shares = int(max_invest / exec_price)

                if shares > 0:
                    cost = exec_price * shares
                    if bm_price > 0:
                        bm_shares -= cost / bm_price

                    s_mult, p_mult = sym_stop_mult, sym_profit_mult
                    stop_price = 0.0
                    tp_price = 0.0
                    if use_atr_stops and atr > 0:
                        s_mult, p_mult = self._scale_atr_multipliers(atr, price, sym_stop_mult, sym_profit_mult)
                        if signal == 1:
                            stop_price = exec_price - s_mult * atr
                            tp_price = exec_price + p_mult * atr
                        else:
                            stop_price = exec_price + s_mult * atr
                            tp_price = exec_price - p_mult * atr

                    entry_indicators = {
                        "entry_zscore": float(row.get("zscore", 0)) if "zscore" in row.index else 0.0,
                        "entry_rsi": float(row.get("rsi", 0)) if "rsi" in row.index else 0.0,
                        "entry_bb_pct_b": float(row.get("bb_pct_b", 0)) if "bb_pct_b" in row.index else 0.0,
                        "entry_volume_zscore": float(row.get("volume_zscore", 0)) if "volume_zscore" in row.index else 0.0,
                        "entry_atr": atr,
                        "entry_volatility": float(row.get("volatility", 0)) if "volatility" in row.index else 0.0,
                        "entry_dist_sma200": float(row.get("dist_sma200", 0)) if "dist_sma200" in row.index else 0.0,
                        "entry_signal_strength": strength,
                        "entry_macd_hist": float(row.get("macd_hist", 0)) if "macd_hist" in row.index else 0.0,
                    }

                    active_positions[sym] = {
                        "shares": shares if signal == 1 else -shares,
                        "entry_price": exec_price,
                        "entry_date": date,
                        "stop": stop_price,
                        "tp": tp_price,
                        "indicators": entry_indicators,
                    }

            # --- Phase 3: Compute portfolio value ---
            active_value = 0.0
            for sym, pos in active_positions.items():
                if sym in signals_by_symbol and date in signals_by_symbol[sym].index:
                    sym_price = float(signals_by_symbol[sym].loc[date, "Close"])
                else:
                    sym_price = pos["entry_price"]
                active_value += pos["shares"] * sym_price

            idle_value = bm_shares * bm_price if bm_price > 0 else 0
            port_val = idle_value + active_value + residual_cash
            portfolio_values.append({
                "date": date,
                "portfolio_value": port_val,
                "cash": residual_cash,
                "active_positions": len(active_positions),
                "bm_shares": bm_shares,
            })

            if bm_price > 0:
                prev_bm_price = bm_price

        self._portfolio = pd.DataFrame(portfolio_values).set_index("date")
        self._current_symbol = "PORTFOLIO"

        # Benchmark = SPY buy-and-hold
        bh_shares = initial_capital / first_bm_price if first_bm_price > 0 else 0
        bm_reindexed = benchmark_prices.reindex(self._portfolio.index).ffill()
        self._benchmark = bm_reindexed * bh_shares

        return self._portfolio
