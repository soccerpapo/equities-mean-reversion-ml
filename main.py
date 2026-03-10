#!/usr/bin/env python3
"""Main CLI orchestrator for the equities mean reversion ML trading system."""

import argparse
import logging
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received. Exiting gracefully...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def run_backtest(symbols: list, years: int = 2) -> None:
    """Fetch data, compute indicators, generate signals, ML filter, run backtest.

    Runs backtest on each symbol individually and prints a summary table with
    results for each symbol plus aggregate portfolio performance.

    Args:
        symbols: List of ticker symbols to backtest
        years: Number of years of data to use for backtesting
    """
    from data.fetcher import DataFetcher
    from features.indicators import IndicatorEngine
    from strategy.signals import SignalGenerator
    from strategy.ml_filter import MLSignalFilter
    from backtest.engine import BacktestEngine
    from config import settings

    period = f"{years}y"
    fetcher = DataFetcher()
    ind_engine = IndicatorEngine()
    sig_gen = SignalGenerator()

    summary_rows = []

    for symbol in symbols:
        logger.info(f"Starting backtest for {symbol}...")
        df = fetcher.fetch_historical(symbol, period=period)
        if df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            continue

        df = ind_engine.compute_all(df)
        df = sig_gen.generate_mean_reversion_signals(df)

        ml_filter = MLSignalFilter()
        ml_filter.train(df)
        df = ml_filter.filter_signals(df)

        bt = BacktestEngine()
        bt.run(df, df)
        report = bt.get_performance_report()
        bt.plot_results(output_dir=".")

        report["symbol"] = symbol
        summary_rows.append(report)

        print(f"\n=== Backtest Report: {symbol} ===")
        for k, v in report.items():
            if k != "symbol":
                print(f"  {k}: {v}")

    if len(summary_rows) > 1:
        print("\n=== Summary Table ===")
        cols = ["symbol", "total_return", "sharpe_ratio", "num_trades", "win_rate", "max_drawdown"]
        header = "  ".join(f"{c:>20}" for c in cols)
        print(header)
        print("-" * len(header))
        for row in summary_rows:
            line = "  ".join(f"{str(row.get(c, 'N/A')):>20}" for c in cols)
            print(line)

        # Aggregate portfolio performance (equal-weight average of returns)
        total_returns = [r["total_return"] for r in summary_rows if "total_return" in r]
        sharpes = [r["sharpe_ratio"] for r in summary_rows if "sharpe_ratio" in r]
        total_trades = sum(r.get("num_trades", 0) for r in summary_rows)
        if total_returns:
            avg_return = sum(total_returns) / len(total_returns)
            avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
            print(f"\n=== Aggregate Portfolio Performance ===")
            print(f"  avg_total_return: {avg_return:.4f}")
            print(f"  avg_sharpe_ratio: {avg_sharpe:.4f}")
            print(f"  total_trades_all_symbols: {total_trades}")


def run_train(symbols: list, years: int = 5) -> None:
    """Train ML model on multiple symbols combined and save it.

    Args:
        symbols: List of ticker symbols to train on
        years: Number of years of data to use for training
    """
    import os
    from strategy.ml_filter import MLSignalFilter
    from config import settings

    logger.info(f"Training ML model on {len(symbols)} symbols: {symbols}")

    ml_filter = MLSignalFilter()
    ml_filter.train_multi_symbol(symbols)

    os.makedirs("models", exist_ok=True)
    model_name = "multi_symbol_model" if len(symbols) > 1 else f"{symbols[0]}_model"
    ml_filter.save_model(f"models/{model_name}.joblib")
    logger.info("Training complete.")


def run_trade() -> None:
    """Live paper trading loop."""
    from config import settings
    from data.fetcher import DataFetcher
    from features.indicators import IndicatorEngine
    from strategy.signals import SignalGenerator
    from risk.manager import RiskManager
    from execution.trader import AlpacaTrader

    trader = AlpacaTrader()
    fetcher = DataFetcher()
    ind_engine = IndicatorEngine()
    sig_gen = SignalGenerator()
    risk_mgr = RiskManager()

    logger.info("Starting live trading loop. Press Ctrl+C to stop.")
    while _running:
        try:
            if not trader.is_market_open():
                logger.info("Market closed. Sleeping 60s...")
                time.sleep(60)
                continue

            acct = trader.get_account()
            if acct is None:
                logger.warning("Could not get account info. Sleeping...")
                time.sleep(settings.TRADING_INTERVAL_SECONDS)
                continue

            account_value = acct["equity"]
            positions = {p["symbol"]: p for p in trader.get_positions()}

            for symbol in settings.SYMBOLS:
                if not _running:
                    break
                df = fetcher.fetch_historical(symbol, period="6mo")
                if df.empty:
                    continue
                df = ind_engine.compute_all(df)
                df = sig_gen.generate_mean_reversion_signals(df)

                latest = df.iloc[-1]
                signal = int(latest.get("signal", 0))
                strength = float(latest.get("signal_strength", 0))
                price = latest["Close"]

                if signal == 1 and symbol not in positions:
                    qty = risk_mgr.calculate_position_size(account_value, price, strength)
                    if qty > 0:
                        tp = price * (1 + settings.TAKE_PROFIT_PCT)
                        sl = price * (1 - settings.STOP_LOSS_PCT)
                        trader.place_bracket_order(symbol, qty, "buy", tp, sl)

                elif signal == -1 and symbol in positions:
                    trader.close_position(symbol)

        except Exception as e:
            logger.error(f"Error in trading loop: {e}")

        time.sleep(settings.TRADING_INTERVAL_SECONDS)

    logger.info("Trading loop stopped.")


def main():
    """Entry point for the trading system CLI."""
    from config import settings

    parser = argparse.ArgumentParser(description="Equities Mean Reversion ML Trading System")
    parser.add_argument(
        "--mode",
        choices=["backtest", "train", "trade"],
        default="backtest",
        help="Operation mode",
    )
    parser.add_argument("--symbol", default="SPY", help="Single symbol (backtest/train modes, overridden by --symbols)")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override default symbols list (e.g. --symbols SPY AAPL MSFT)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Number of years of data to use for training/backtesting",
    )
    args = parser.parse_args()

    if args.mode == "backtest":
        symbols = args.symbols or [args.symbol]
        years = args.years or 2
        run_backtest(symbols, years=years)
    elif args.mode == "train":
        symbols = args.symbols or settings.TRAINING_SYMBOLS
        years = args.years or getattr(settings, "ML_LOOKBACK_YEARS", 5)
        run_train(symbols, years=years)
    elif args.mode == "trade":
        run_trade()


if __name__ == "__main__":
    main()
