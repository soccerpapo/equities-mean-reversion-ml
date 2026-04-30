import os
import logging
import numpy as np
import pandas as pd
import itertools
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from data.fetcher import DataFetcher
from features.indicators import IndicatorEngine
from strategy.signals import SignalGenerator
from strategy.pairs_trading import PairsTrader
from strategy.momentum import MomentumTrader
from strategy.regime_detector import RegimeDetector
from backtest.engine import BacktestEngine
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def run_regime_optimization(train_years=5):
    """Walk-Forward Regime Optimizer
    1. Fetches training data.
    2. Generates daily returns for 3 baseline strategies: Mean-Reversion, Pairs, Momentum.
    3. Detects the GMM regime for each day.
    4. Finds the optimal allocation for each regime to maximize Sharpe Ratio.
    """
    end_date = pd.to_datetime(getattr(settings, "BACKTEST_END_DATE", datetime.today().strftime("%Y-%m-%d")))
    # We backtest on the last 2 years, so training data should end 2 years ago.
    train_end_dt = end_date - relativedelta(years=2)
    train_start_dt = train_end_dt - relativedelta(years=train_years)

    start_str = train_start_dt.strftime("%Y-%m-%d")
    end_str = train_end_dt.strftime("%Y-%m-%d")
    
    logger.info(f"Optimizing regimes on training data from {start_str} to {end_str}...")

    symbols = settings.SYMBOLS
    fetcher = DataFetcher()
    ind_engine = IndicatorEngine()
    sig_gen = SignalGenerator()

    # 1. Fetch data
    all_data = {}
    for sym in symbols + ["SPY"]:
        df = fetcher.fetch_historical(sym, start_date=start_str, end_date=end_str)
        if not df.empty:
            all_data[sym] = df

    if "SPY" not in all_data:
        logger.error("SPY data missing.")
        return

    df_spy = ind_engine.compute_all(all_data["SPY"])
    
    # 2. Regime Detection
    detector = RegimeDetector(n_components=getattr(settings, "REGIME_N_COMPONENTS", 3))
    detector.fit(df_spy)
    df_regime = df_spy.copy()
    regime_series = pd.Series(index=df_spy.index, dtype=int)
    for idx in df_spy.index:
        r, _ = detector.detect_regime(df_spy.loc[:idx])
        regime_series[idx] = r

    # 3. Strategy 1: Mean Reversion (Daily Returns)
    logger.info("Running Mean-Reversion backtest on training data...")
    mr_data = {sym: df for sym, df in all_data.items() if sym != "SPY"}
    bt_mr = BacktestEngine()
    df_base = pd.DataFrame() # Just need one to run - wait, BacktestEngine.run takes df_signals
    # We need to run it properly. Actually BacktestEngine.run_portfolio_backtest runs it for all.
    # Let's generate signals for all mr_data
    for sym, df in mr_data.items():
        df_ind = ind_engine.compute_all(df)
        mr_data[sym] = sig_gen.generate_mean_reversion_signals(df_ind)
    
    # We can just use the portfolio backtest method. Wait, run_portfolio isn't in BacktestEngine, it's in main.py?
    # No, BacktestEngine.run() takes single symbol.
    # Let's just run an aggregate portfolio approximation. Or simply run the individual BacktestEngines.
    # Wait, `run_portfolio_backtest` is not in BacktestEngine? Let's check `backtest/engine.py`.
    # Let's use a simpler approach: get daily returns from run_momentum_backtest and run_pairs_backtest.
    logger.info("Running Pairs backtest...")
    pt = PairsTrader()
    price_series = {sym: df["Close"] for sym, df in mr_data.items()}
    coint_pairs = pt.find_cointegrated_pairs(list(mr_data.keys()), price_series)
    pairs_input = []
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
    bt_pairs = BacktestEngine()
    df_pairs_port = bt_pairs.run_pairs_backtest(pairs_input, initial_capital=100000.0)

    logger.info("Running Momentum backtest...")
    bt_mom = BacktestEngine()
    df_mom_port = bt_mom.run_momentum_backtest(mr_data, initial_capital=100000.0)

    logger.info("Running Mean-Reversion portfolio backtest (approximated)...")
    # Since we need a portfolio for MR, we can sum up individual backtests
    mr_port_val = pd.Series(100000.0, index=df_spy.index)
    cap_per_sym = 100000.0 / len(mr_data)
    mr_total = pd.Series(0.0, index=df_spy.index)
    for sym, df in mr_data.items():
        bt = BacktestEngine()
        df_port = bt.run(df, df, initial_capital=cap_per_sym)
        if not df_port.empty:
            # align to spy index
            aligned = df_port["portfolio_value"].reindex(df_spy.index).ffill()
            mr_total += aligned.fillna(cap_per_sym)
        else:
            mr_total += cap_per_sym

    # Get daily returns
    mr_returns = mr_total.pct_change().fillna(0)
    pairs_returns = df_pairs_port["portfolio_value"].pct_change().reindex(df_spy.index).fillna(0) if not df_pairs_port.empty else pd.Series(0, index=df_spy.index)
    mom_returns = df_mom_port["portfolio_value"].pct_change().reindex(df_spy.index).fillna(0) if not df_mom_port.empty else pd.Series(0, index=df_spy.index)

    # Create a DataFrame
    df_all = pd.DataFrame({
        "regime": regime_series,
        "mr": mr_returns,
        "pairs": pairs_returns,
        "mom": mom_returns,
        "cash": 0.0 # cash earns 0 daily return for simplicity
    }).dropna()

    # Optimization
    allocations = [
        (mr, p, mo, c)
        for mr in np.arange(0, 1.1, 0.1)
        for p in np.arange(0, 1.1, 0.1)
        for mo in np.arange(0, 1.1, 0.1)
        for c in np.arange(0, 1.1, 0.1)
        if np.isclose(mr + p + mo + c, 1.0)
    ]

    optimal_allocations = {}

    for regime_id in [0, 1, 2]:
        df_r = df_all[df_all["regime"] == regime_id]
        if len(df_r) < 20:
            logger.warning(f"Not enough data for regime {regime_id}. Using default.")
            optimal_allocations[regime_id] = (0.25, 0.25, 0.25, 0.25)
            continue
        
        best_sharpe = -999
        best_alloc = None
        
        for w_mr, w_p, w_mo, w_c in allocations:
            blended_returns = (df_r["mr"] * w_mr + df_r["pairs"] * w_p + df_r["mom"] * w_mo + df_r["cash"] * w_c)
            std = blended_returns.std()
            if std > 0:
                sharpe = (blended_returns.mean() / std) * np.sqrt(252)
            else:
                sharpe = 0
                
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_alloc = (w_mr, w_p, w_mo, w_c)
                
        optimal_allocations[regime_id] = best_alloc
        logger.info(f"Regime {regime_id} optimal alloc: MR={best_alloc[0]:.1f}, Pairs={best_alloc[1]:.1f}, Mom={best_alloc[2]:.1f}, Cash={best_alloc[3]:.1f} (Sharpe: {best_sharpe:.2f})")

    print("\n=== OPTIMAL REGIME ALLOCATIONS ===")
    for r, alloc in optimal_allocations.items():
        print(f"Regime {r}: {{'mean_reversion': {alloc[0]:.1f}, 'pairs': {alloc[1]:.1f}, 'momentum': {alloc[2]:.1f}, 'cash': {alloc[3]:.1f}}}")

if __name__ == "__main__":
    run_regime_optimization()
