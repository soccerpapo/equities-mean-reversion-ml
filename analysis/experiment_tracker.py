"""Simple CSV-based experiment tracker for parameter tuning."""

import csv
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

EXPERIMENT_LOG_PATH = "experiments/experiment_log.csv"

COLUMNS = [
    "timestamp",
    "experiment_id",
    "symbols",
    "years",
    "strategy",
    # Parameters
    "z_score_entry",
    "min_signal_strength",
    "rsi_oversold",
    "use_trend_filter",
    "use_vol_filter",
    "use_dist_sma200_filter",
    "max_dist_sma200",
    "atr_stop_mult",
    "atr_profit_mult",
    "stop_loss_pct",
    "take_profit_pct",
    "min_optional_confirmations",
    # Results
    "total_return",
    "annualized_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "num_trades",
    "win_rate",
    "profit_factor",
    "avg_win",
    "avg_loss",
    "benchmark_return",
    "alpha",
    # Trade analysis
    "stop_hit_rate",
    "tp_hit_rate",
    "expectancy",
    "avg_holding_days",
    # Notes
    "notes",
]


def _ensure_log_exists() -> None:
    os.makedirs(os.path.dirname(EXPERIMENT_LOG_PATH), exist_ok=True)
    if not os.path.exists(EXPERIMENT_LOG_PATH):
        with open(EXPERIMENT_LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()


def log_experiment(
    symbols: list,
    years: int,
    strategy: str,
    report: Dict,
    trade_analysis: Optional[Dict] = None,
    notes: str = "",
) -> str:
    """Log one experiment run to the CSV tracker.

    Args:
        symbols: Symbols tested
        years: Years of data
        strategy: Strategy name
        report: Performance report dict from BacktestEngine
        trade_analysis: Optional trade analysis dict
        notes: Free-text notes

    Returns:
        The experiment_id assigned
    """
    from config import settings

    _ensure_log_exists()

    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ta = trade_analysis or {}

    row = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": exp_id,
        "symbols": ",".join(symbols),
        "years": years,
        "strategy": strategy,
        "z_score_entry": getattr(settings, "Z_SCORE_ENTRY_THRESHOLD", ""),
        "min_signal_strength": getattr(settings, "MIN_SIGNAL_STRENGTH", ""),
        "rsi_oversold": getattr(settings, "RSI_OVERSOLD", ""),
        "use_trend_filter": getattr(settings, "USE_TREND_FILTER", ""),
        "use_vol_filter": getattr(settings, "USE_VOLATILITY_FILTER", ""),
        "use_dist_sma200_filter": getattr(settings, "USE_DIST_SMA200_FILTER", ""),
        "max_dist_sma200": getattr(settings, "MAX_DIST_SMA200", ""),
        "atr_stop_mult": getattr(settings, "ATR_STOP_MULTIPLIER", ""),
        "atr_profit_mult": getattr(settings, "ATR_PROFIT_MULTIPLIER", ""),
        "stop_loss_pct": getattr(settings, "STOP_LOSS_PCT", ""),
        "take_profit_pct": getattr(settings, "TAKE_PROFIT_PCT", ""),
        "min_optional_confirmations": getattr(settings, "MIN_OPTIONAL_CONFIRMATIONS", ""),
        "total_return": report.get("total_return", ""),
        "annualized_return": report.get("annualized_return", ""),
        "sharpe_ratio": report.get("sharpe_ratio", ""),
        "sortino_ratio": report.get("sortino_ratio", ""),
        "max_drawdown": report.get("max_drawdown", ""),
        "num_trades": report.get("num_trades", ""),
        "win_rate": report.get("win_rate", ""),
        "profit_factor": report.get("profit_factor", ""),
        "avg_win": report.get("avg_win", ""),
        "avg_loss": report.get("avg_loss", ""),
        "benchmark_return": report.get("benchmark_return", ""),
        "alpha": report.get("alpha", ""),
        "stop_hit_rate": ta.get("stop_hit_rate", ""),
        "tp_hit_rate": ta.get("tp_hit_rate", ""),
        "expectancy": ta.get("expectancy_per_trade", ""),
        "avg_holding_days": ta.get("avg_holding_days", ""),
        "notes": notes,
    }

    with open(EXPERIMENT_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(row)

    logger.info(f"Experiment {exp_id} logged to {EXPERIMENT_LOG_PATH}")
    return exp_id


def load_experiments() -> pd.DataFrame:
    """Load the full experiment log as a DataFrame.

    Returns:
        DataFrame of all logged experiments
    """
    _ensure_log_exists()
    return pd.read_csv(EXPERIMENT_LOG_PATH)


def print_experiment_summary() -> None:
    """Print a summary of all experiments sorted by Sharpe ratio."""
    df = load_experiments()
    if df.empty:
        print("No experiments logged yet.")
        return

    display_cols = [
        "experiment_id", "strategy", "symbols",
        "total_return", "sharpe_ratio", "max_drawdown",
        "num_trades", "win_rate", "alpha", "z_score_entry",
        "min_signal_strength",
    ]
    available = [c for c in display_cols if c in df.columns]
    summary = df[available].copy()
    if "sharpe_ratio" in summary.columns:
        summary = summary.sort_values("sharpe_ratio", ascending=False)
    print("\n=== Experiment Tracker Summary ===")
    print(summary.to_string(index=False))
