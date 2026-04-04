import pandas as pd
import logging
from typing import Dict
from config import settings
from main import run_portfolio
from data.fetcher import DataFetcher
from strategy.signals import SignalGenerator
from backtest.engine import BacktestEngine
from analysis.stock_profiles import calibrate_all

# Suppress info logs
logging.getLogger("strategy.signals").setLevel(logging.WARNING)
logging.getLogger("strategy.ml_filter").setLevel(logging.WARNING)
logging.getLogger("data.fetcher").setLevel(logging.WARNING)
logging.getLogger("__main__").setLevel(logging.WARNING)
logging.getLogger("backtest.engine").setLevel(logging.WARNING)

def test_config(name, config_updates):
    # Save original settings
    original = {}
    for k, v in config_updates.items():
        original[k] = getattr(settings, k)
        setattr(settings, k, v)
        
    print(f"\n--- Testing: {name} ---")
    
    try:
        symbols = getattr(settings, "SYMBOLS", ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"])
        # Suppress printing the comparison table from run_portfolio
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        # Run
        run_portfolio(symbols, years=2)
        
        # Get output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Extract metrics
        metrics = {}
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith("total_return:"): metrics["Return"] = line.split(":")[1].strip()
            if line.startswith("sharpe_ratio:"): metrics["Sharpe"] = line.split(":")[1].strip()
            if line.startswith("num_trades:"): metrics["Trades"] = line.split(":")[1].strip()
            if line.startswith("win_rate:"): metrics["Win Rate"] = line.split(":")[1].strip()
            if line.startswith("alpha:"): metrics["Alpha"] = line.split(":")[1].strip()
            
        print(f"Trades: {metrics.get('Trades', 'N/A')} | Return: {metrics.get('Return', 'N/A')} | Sharpe: {metrics.get('Sharpe', 'N/A')} | Win Rate: {metrics.get('Win Rate', 'N/A')} | Alpha: {metrics.get('Alpha', 'N/A')}")
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Error: {e}")
        
    # Restore original settings
    for k, v in original.items():
        setattr(settings, k, v)

if __name__ == "__main__":
    configs = [
        ("Baseline (Current)", {}),
        ("Disable LONG_ONLY (Allow Shorts)", {"LONG_ONLY": False}),
        ("Disable Trend Filter", {"USE_TREND_FILTER": False}),
        ("Disable Volatility Filter", {"USE_VOLATILITY_FILTER": False}),
        ("Disable Dist-SMA200 Filter", {"USE_DIST_SMA200_FILTER": False}),
        ("Lower Entry Threshold to 1.5", {"Z_SCORE_ENTRY_THRESHOLD": 1.5}),
        ("Lower Entry to 1.5 & Allow Shorts", {"Z_SCORE_ENTRY_THRESHOLD": 1.5, "LONG_ONLY": False}),
    ]
    
    for name, updates in configs:
        test_config(name, updates)
