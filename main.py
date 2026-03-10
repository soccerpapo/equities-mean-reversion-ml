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


def run_backtest(symbol: str = "SPY") -> None:
    """Fetch data, compute indicators, generate signals, ML filter, run backtest.

    Args:
        symbol: Ticker symbol to backtest
    """
    from data.fetcher import DataFetcher
    from features.indicators import IndicatorEngine
    from strategy.signals import SignalGenerator
    from strategy.ml_filter import MLSignalFilter
    from backtest.engine import BacktestEngine

    logger.info(f"Starting backtest for {symbol}...")
    fetcher = DataFetcher()
    df = fetcher.fetch_historical(symbol, period="2y")
    if df.empty:
        logger.error("Failed to fetch data")
        return

    engine = IndicatorEngine()
    df = engine.compute_all(df)

    sig_gen = SignalGenerator()
    df = sig_gen.generate_mean_reversion_signals(df)

    ml_filter = MLSignalFilter()
    ml_filter.train(df)
    df = ml_filter.filter_signals(df)

    bt = BacktestEngine()
    bt.run(df, df)
    report = bt.get_performance_report()
    bt.plot_results()

    print("\n=== Backtest Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")


def run_train(symbol: str = "SPY") -> None:
    """Train ML model and save it.

    Args:
        symbol: Ticker symbol to train on
    """
    import os
    from data.fetcher import DataFetcher
    from features.indicators import IndicatorEngine
    from strategy.ml_filter import MLSignalFilter

    logger.info(f"Training ML model on {symbol}...")
    fetcher = DataFetcher()
    df = fetcher.fetch_historical(symbol, period="5y")
    if df.empty:
        logger.error("Failed to fetch data")
        return

    engine = IndicatorEngine()
    df = engine.compute_all(df)

    ml_filter = MLSignalFilter()
    ml_filter.train(df)

    os.makedirs("models", exist_ok=True)
    ml_filter.save_model(f"models/{symbol}_model.joblib")
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
    parser = argparse.ArgumentParser(description="Equities Mean Reversion ML Trading System")
    parser.add_argument(
        "--mode",
        choices=["backtest", "train", "trade"],
        default="backtest",
        help="Operation mode",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol to use (backtest/train modes)")
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(args.symbol)
    elif args.mode == "train":
        run_train(args.symbol)
    elif args.mode == "trade":
        run_trade()


if __name__ == "__main__":
    main()
