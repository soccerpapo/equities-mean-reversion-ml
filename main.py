#!/usr/bin/env python3
"""Main CLI orchestrator for the equities mean reversion ML trading system."""

import argparse
import logging
import signal
import sys
import time

import pandas as pd

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


def _apply_regime_labels(df: pd.DataFrame, detector) -> pd.Series:
    """Compute per-row regime labels and apply position scaling to a signals DataFrame.

    This avoids duplicating the regime-labeling loop across run_backtest,
    run_compare, and _prepare_signals.

    Args:
        df: Signals DataFrame (will be modified in-place for signal/signal_strength).
        detector: A fitted RegimeDetector instance.

    Returns:
        Series of regime labels (int, 0/1/2) aligned to df.index.
    """
    labels = []
    for i in range(len(df)):
        if i < 30:
            labels.append(1)  # default to normal regime for early rows
        else:
            lbl, _ = detector.detect_regime(df.iloc[: i + 1])
            labels.append(lbl)
    regime_series = pd.Series(labels, index=df.index, dtype=int)
    multipliers = regime_series.map(detector.get_position_multiplier)
    df["signal_strength"] = df["signal_strength"] * multipliers
    df.loc[regime_series == 2, "signal"] = 0
    df.loc[regime_series == 2, "signal_strength"] = 0.0
    logger.info("Regime distribution: %s", regime_series.value_counts().to_dict())
    return regime_series


def _prepare_signals(
    symbol: str,
    period: str,
    fetcher,
    ind_engine,
    sig_gen,
    use_ml: bool = True,
    use_regime: bool = False,
):
    """Fetch data and return (signals_df, regime_series).

    Args:
        symbol: Ticker symbol
        period: yfinance period string (e.g. '2y')
        fetcher: DataFetcher instance
        ind_engine: IndicatorEngine instance
        sig_gen: SignalGenerator instance
        use_ml: Whether to apply the ML signal filter
        use_regime: Whether to compute and return regime labels

    Returns:
        Tuple (signals_df, regime_series) where regime_series may be None
    """
    from strategy.ml_filter import MLSignalFilter
    from strategy.regime_detector import RegimeDetector
    from config import settings

    df = fetcher.fetch_historical(symbol, period=period)
    if df.empty:
        logger.error(f"Failed to fetch data for {symbol}")
        return None, None

    df = ind_engine.compute_all(df)
    df = sig_gen.generate_mean_reversion_signals(df)

    if use_ml:
        ml_filter = MLSignalFilter()
        ml_filter.train(df)
        df = ml_filter.filter_signals(df)

    regime_series = None
    if use_regime:
        n_components = getattr(settings, "REGIME_N_COMPONENTS", 3)
        detector = RegimeDetector(n_components=n_components)
        detector.fit(df)
        regime_series = _apply_regime_labels(df, detector)

    return df, regime_series


def run_backtest(
    symbols: list,
    years: int = 2,
    use_ml: bool = True,
    use_regime: bool = False,
    use_atr_stops: bool = False,
) -> None:
    """Fetch data, compute indicators, generate signals, ML filter, run backtest.

    Args:
        symbols: List of ticker symbols to backtest
        years: Number of years of data to use for backtesting
        use_ml: Whether to apply the ML signal filter (False for pure statistical)
        use_regime: Whether to use regime detection for position sizing
        use_atr_stops: Whether to use ATR-based dynamic stops
    """
    import pandas as pd
    from data.fetcher import DataFetcher
    from features.indicators import IndicatorEngine
    from strategy.signals import SignalGenerator
    from backtest.engine import BacktestEngine
    from config import settings

    period = f"{years}y"
    fetcher = DataFetcher()
    ind_engine = IndicatorEngine()
    sig_gen = SignalGenerator()

    summary_rows = []

    for symbol in symbols:
        logger.info(f"Starting backtest for {symbol}...")
        df, regime_series = _prepare_signals(
            symbol, period, fetcher, ind_engine, sig_gen,
            use_ml=use_ml, use_regime=use_regime,
        )
        if df is None:
            continue

        bt = BacktestEngine()
        bt.run(
            df,
            df,
            use_atr_stops=use_atr_stops,
            atr_stop_mult=getattr(settings, "ATR_STOP_MULTIPLIER", 2.0),
            atr_profit_mult=getattr(settings, "ATR_PROFIT_MULTIPLIER", 3.0),
        )
        report = bt.get_performance_report()
        bt.plot_results(output_dir=".", regime_series=regime_series)

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


def run_compare(symbols: list, years: int = 2) -> None:
    """Run all six approaches on the same data and print a comparison table.

    Approaches:
      1. Pure Statistical (no ML, no regime detection, fixed stops)
      2. + Regime Detection (no ML, with regime detection, ATR stops)
      3. + ML Filter (with ML, no regime detection, fixed stops)
      4. Pairs Trading
      5. Momentum
      6. Adaptive (Regime Switch)

    Args:
        symbols: List of ticker symbols to test
        years: Number of years of historical data to use
    """
    import pandas as pd
    from data.fetcher import DataFetcher
    from features.indicators import IndicatorEngine
    from strategy.signals import SignalGenerator
    from strategy.ml_filter import MLSignalFilter
    from strategy.regime_detector import RegimeDetector
    from strategy.pairs_trading import PairsTrader
    from backtest.engine import BacktestEngine
    from config import settings

    period = f"{years}y"
    fetcher = DataFetcher()
    ind_engine = IndicatorEngine()
    sig_gen = SignalGenerator()

    comparison_rows = []

    # Fetch all data once
    all_data = {}
    for sym in symbols:
        df = fetcher.fetch_historical(sym, period=period)
        if not df.empty:
            all_data[sym] = df

    for symbol in symbols:
        if symbol not in all_data:
            logger.error(f"Failed to fetch data for {symbol}")
            continue

        logger.info(f"Running comparison for {symbol}...")
        df_raw = all_data[symbol]
        df_base = ind_engine.compute_all(df_raw)
        df_base = sig_gen.generate_mean_reversion_signals(df_base)

        # ---- Path A: Pure Statistical ----
        bt_a = BacktestEngine()
        bt_a.run(df_base, df_base)
        report_a = bt_a.get_performance_report()
        report_a["approach"] = f"{symbol}: Pure Statistical"
        comparison_rows.append(report_a)

        # ---- Path B: + Regime Detection ----
        n_components = getattr(settings, "REGIME_N_COMPONENTS", 3)
        detector = RegimeDetector(n_components=n_components)
        detector.fit(df_base)
        df_regime = df_base.copy()
        regime_series = _apply_regime_labels(df_regime, detector)

        bt_b = BacktestEngine()
        bt_b.run(
            df_regime,
            df_regime,
            use_atr_stops=True,
            atr_stop_mult=getattr(settings, "ATR_STOP_MULTIPLIER", 2.0),
            atr_profit_mult=getattr(settings, "ATR_PROFIT_MULTIPLIER", 3.0),
        )
        report_b = bt_b.get_performance_report()
        report_b["approach"] = f"{symbol}: + Regime Detection"
        comparison_rows.append(report_b)

        # ---- Path C: + ML Filter ----
        df_ml = df_base.copy()
        ml_filter = MLSignalFilter()
        ml_filter.train(df_ml)
        df_ml = ml_filter.filter_signals(df_ml)

        bt_c = BacktestEngine()
        bt_c.run(df_ml, df_ml)
        report_c = bt_c.get_performance_report()
        report_c["approach"] = f"{symbol}: + ML Filter"
        comparison_rows.append(report_c)

        # ---- Buy & Hold (per-symbol benchmark) ----
        bh_return = report_a.get("benchmark_return", float("nan"))
        comparison_rows.append({
            "approach": f"{symbol}: Buy & Hold SPY",
            "total_return": bh_return,
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "num_trades": 1,
            "win_rate": float("nan"),
        })

    # ---- Path D: Pairs Trading (cross-symbol, run once) ----
    if len(all_data) >= 2:
        pt = PairsTrader()
        price_series = {s: all_data[s]["Close"] for s in all_data}
        coint_pairs = pt.find_cointegrated_pairs(list(all_data.keys()), price_series)
        if coint_pairs:
            pairs_input = [
                {
                    "symbol_a": pa, "symbol_b": pb,
                    "price_a": price_series[pa], "price_b": price_series[pb],
                    "hedge_ratio": hr,
                }
                for pa, pb, _, _, hr in coint_pairs
                if pa in price_series and pb in price_series
            ]
            bt_d = BacktestEngine()
            bt_d.run_pairs_backtest(pairs_input, initial_capital=100_000.0)
            report_d = bt_d.get_performance_report()
            report_d["approach"] = "Pairs Trading"
            comparison_rows.append(report_d)

    # ---- Path E: Momentum (cross-symbol, run once) ----
    bt_e = BacktestEngine()
    bt_e.run_momentum_backtest(all_data, initial_capital=100_000.0)
    report_e = bt_e.get_performance_report()
    report_e["approach"] = "Momentum"
    comparison_rows.append(report_e)

    # ---- Path F: Adaptive (cross-symbol, run once) ----
    bt_f = BacktestEngine()
    bt_f.run_adaptive_backtest(all_data, initial_capital=100_000.0)
    report_f = bt_f.get_performance_report()
    report_f["approach"] = "Adaptive (Regime Switch)"
    comparison_rows.append(report_f)

    BacktestEngine.print_comparison_table(comparison_rows)


def run_pairs_backtest(symbols: list, years: int = 2) -> None:
    """Run pairs trading backtest.

    Args:
        symbols: List of ticker symbols to test for cointegration.
        years: Number of years of historical data.
    """
    from data.fetcher import DataFetcher
    from strategy.pairs_trading import PairsTrader
    from backtest.engine import BacktestEngine

    period = f"{years}y"
    fetcher = DataFetcher()
    data_dict = {}
    for sym in symbols:
        df = fetcher.fetch_historical(sym, period=period)
        if not df.empty:
            data_dict[sym] = df

    if not data_dict:
        logger.error("No data fetched for pairs backtest.")
        return

    pt = PairsTrader()
    price_series = {sym: df["Close"] for sym, df in data_dict.items()}
    coint_pairs = pt.find_cointegrated_pairs(list(data_dict.keys()), price_series)

    if not coint_pairs:
        logger.warning("No cointegrated pairs found. Try more symbols or a longer period.")
        return

    pairs_input = [
        {
            "symbol_a": sym_a, "symbol_b": sym_b,
            "price_a": price_series[sym_a], "price_b": price_series[sym_b],
            "hedge_ratio": hedge,
        }
        for sym_a, sym_b, p_val, corr, hedge in coint_pairs
        if sym_a in price_series and sym_b in price_series
    ]

    bt = BacktestEngine()
    bt.run_pairs_backtest(pairs_input, initial_capital=100_000.0)
    report = bt.get_performance_report()
    bt.plot_results(output_dir=".")

    print("\n=== Pairs Trading Backtest Report ===")
    print(f"  Pairs tested: {len(coint_pairs)}")
    for pair in coint_pairs:
        print(f"  {pair[0]}/{pair[1]}: p={pair[2]:.4f} corr={pair[3]:.3f} hedge={pair[4]:.4f}")
    for k, v in report.items():
        print(f"  {k}: {v}")


def run_momentum_backtest(symbols: list, years: int = 2) -> None:
    """Run momentum strategy backtest.

    Args:
        symbols: List of ticker symbols to trade.
        years: Number of years of historical data.
    """
    from data.fetcher import DataFetcher
    from backtest.engine import BacktestEngine

    period = f"{years}y"
    fetcher = DataFetcher()
    symbols_data = {}
    for sym in symbols:
        df = fetcher.fetch_historical(sym, period=period)
        if not df.empty:
            symbols_data[sym] = df

    if not symbols_data:
        logger.error("No data fetched for momentum backtest.")
        return

    bt = BacktestEngine()
    bt.run_momentum_backtest(symbols_data, initial_capital=100_000.0)
    report = bt.get_performance_report()
    bt.plot_results(output_dir=".")

    print("\n=== Momentum Backtest Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")


def run_adaptive_backtest(symbols: list, years: int = 2) -> None:
    """Run adaptive regime-switching strategy backtest.

    Args:
        symbols: List of ticker symbols to trade.
        years: Number of years of historical data.
    """
    from data.fetcher import DataFetcher
    from backtest.engine import BacktestEngine

    period = f"{years}y"
    fetcher = DataFetcher()
    symbols_data = {}
    for sym in symbols:
        df = fetcher.fetch_historical(sym, period=period)
        if not df.empty:
            symbols_data[sym] = df

    if not symbols_data:
        logger.error("No data fetched for adaptive backtest.")
        return

    bt = BacktestEngine()
    bt.run_adaptive_backtest(symbols_data, initial_capital=100_000.0)
    report = bt.get_performance_report()
    bt.plot_results(output_dir=".")

    print("\n=== Adaptive (Regime-Switching) Backtest Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")


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


def run_trade(use_ml: bool = True, use_regime: bool = False) -> None:
    """Live paper trading loop.

    Args:
        use_ml: Whether to apply the ML signal filter
        use_regime: Whether to use regime detection for position sizing
    """
    from config import settings
    from data.fetcher import DataFetcher
    from features.indicators import IndicatorEngine
    from strategy.signals import SignalGenerator
    from strategy.regime_detector import RegimeDetector
    from risk.manager import RiskManager
    from execution.trader import AlpacaTrader

    trader = AlpacaTrader()
    fetcher = DataFetcher()
    ind_engine = IndicatorEngine()
    sig_gen = SignalGenerator()
    risk_mgr = RiskManager()

    # Optionally pre-fit regime detector
    regime_detector = None
    if use_regime:
        n_components = getattr(settings, "REGIME_N_COMPONENTS", 3)
        regime_detector = RegimeDetector(n_components=n_components)

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

                if use_ml:
                    from strategy.ml_filter import MLSignalFilter
                    ml_filter = MLSignalFilter()
                    ml_filter.train(df)
                    df = ml_filter.filter_signals(df)

                regime_multiplier = 1.0
                if regime_detector is not None:
                    regime_detector.fit(df)
                    regime, conf = regime_detector.detect_regime(df)
                    regime_multiplier = regime_detector.get_position_multiplier(regime)
                    logger.info(f"{symbol}: regime={regime}, confidence={conf:.2f}, multiplier={regime_multiplier}")

                latest = df.iloc[-1]
                signal = int(latest.get("signal", 0))
                strength = float(latest.get("signal_strength", 0))
                price = latest["Close"]
                atr = float(latest.get("atr", 0.0))

                if signal == 1 and symbol not in positions:
                    qty = risk_mgr.calculate_position_size(
                        account_value, price, strength, regime_multiplier=regime_multiplier
                    )
                    if qty > 0:
                        use_atr = atr > 0
                        if use_atr:
                            sl, tp = risk_mgr.calculate_atr_stops(
                                price, atr, "long",
                                atr_stop_mult=getattr(settings, "ATR_STOP_MULTIPLIER", 2.0),
                                atr_profit_mult=getattr(settings, "ATR_PROFIT_MULTIPLIER", 3.0),
                            )
                        else:
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
    import pandas as pd
    from config import settings

    parser = argparse.ArgumentParser(description="Equities Mean Reversion ML Trading System")
    parser.add_argument(
        "--mode",
        choices=["backtest", "train", "trade", "compare"],
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
    parser.add_argument(
        "--no-ml",
        action="store_true",
        default=False,
        help="Disable ML filter — use pure statistical signals",
    )
    parser.add_argument(
        "--regime",
        action="store_true",
        default=False,
        help="Enable regime detection for position sizing",
    )
    parser.add_argument(
        "--strategy",
        choices=["mean_reversion", "pairs", "momentum", "adaptive", "all"],
        default="all",
        help=(
            "Strategy to run. 'all' runs everything in compare mode. "
            "Use with --mode backtest or --mode trade to select a specific strategy."
        ),
    )
    args = parser.parse_args()

    use_ml = not args.no_ml

    if args.mode == "backtest":
        symbols = args.symbols or settings.SYMBOLS
        years = args.years or 2
        strategy = args.strategy

        if strategy in ("mean_reversion", "all"):
            run_backtest(symbols, years=years, use_ml=use_ml, use_regime=args.regime)
        if strategy in ("pairs", "all"):
            run_pairs_backtest(symbols, years=years)
        if strategy in ("momentum", "all"):
            run_momentum_backtest(symbols, years=years)
        if strategy in ("adaptive", "all"):
            run_adaptive_backtest(symbols, years=years)

    elif args.mode == "compare":
        symbols = args.symbols or settings.SYMBOLS
        years = args.years or 2
        run_compare(symbols, years=years)
    elif args.mode == "train":
        symbols = args.symbols or settings.TRAINING_SYMBOLS
        years = args.years or getattr(settings, "ML_LOOKBACK_YEARS", 5)
        run_train(symbols, years=years)
    elif args.mode == "trade":
        strategy = args.strategy
        if strategy in ("pairs", "momentum", "adaptive"):
            logger.info(f"Paper trading with strategy: {strategy}")
            logger.info(
                "Note: --strategy %s in live trade mode uses the mean reversion loop "
                "as the execution layer. Full live trading for pairs/momentum/adaptive "
                "requires exchange integration beyond the current demo scope.",
                strategy,
            )
        run_trade(use_ml=use_ml, use_regime=(args.regime or strategy == "adaptive"))


if __name__ == "__main__":
    main()
