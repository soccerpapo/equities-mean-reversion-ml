import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from analysis.symbol_screener import _hurst_exponent, _dip_recovery_rate, _compute_beta
import time

def calibrate_expanding_profile(
    symbol: str,
    prices: pd.Series,
    spy_returns: Optional[pd.Series] = None,
    base_atr_stop: float = 1.5,
    base_atr_profit: float = 2.5,
    base_z_entry: float = 1.7,
    base_min_strength: float = 0.28,
    base_max_pos: float = 0.12,
    update_freq: int = 21,
):
    print(f"Calibrating {symbol} with expanding window...")
    t0 = time.time()
    
    returns = prices.pct_change().dropna()
    dates = returns.index
    n = len(dates)
    
    # Store daily params
    daily_params = {
        "atr_stop_mult": np.full(n, base_atr_stop),
        "atr_profit_mult": np.full(n, base_atr_profit),
        "z_score_entry_threshold": np.full(n, base_z_entry),
        "min_signal_strength": np.full(n, base_min_strength),
        "max_position_size_pct": np.full(n, base_max_pos),
        "ann_vol": np.full(n, np.nan),
        "hurst": np.full(n, np.nan),
        "dip_recovery": np.full(n, np.nan),
        "beta": np.full(n, 1.0),
    }
    
    last_params = {k: v[0] for k, v in daily_params.items()}
    
    for i in range(100, n):
        # Update every `update_freq` days
        if i % update_freq == 0:
            window_prices = prices.iloc[:i+1] # Prices up to current day (T) wait, if it's up to T-1, we should use :i
            window_returns = returns.iloc[:i] # Returns up to T-1
            
            # Ann Volatility
            ann_vol = float(window_returns.std() * np.sqrt(252))
            
            # Hurst
            hurst = _hurst_exponent(window_returns)
            
            # Dip recovery
            recovery_rate, _ = _dip_recovery_rate(window_prices, zscore_threshold=-base_z_entry)
            
            # Beta
            beta = 1.0
            if spy_returns is not None:
                beta = _compute_beta(window_returns, spy_returns.iloc[:i])
            
            # Calculations
            vol_scale = np.clip(0.5 + (ann_vol - 0.15) / 0.30 * 1.0, 0.75, 1.6)
            atr_stop = base_atr_stop * vol_scale
            atr_profit = base_atr_profit * vol_scale
            z_entry = base_z_entry + (hurst - 0.50) * 2.5
            z_entry = float(np.clip(z_entry, 1.3, 2.3))
            pos_size = base_max_pos * (1.0 - (ann_vol - 0.25) * 1.5)
            pos_size = float(np.clip(pos_size, 0.05, 0.18))
            strength_adjust = (recovery_rate - 0.80) * 0.6
            min_strength = base_min_strength - strength_adjust
            min_strength = float(np.clip(min_strength, 0.18, 0.45))
            
            last_params.update({
                "atr_stop_mult": round(atr_stop, 3),
                "atr_profit_mult": round(atr_profit, 3),
                "z_score_entry_threshold": round(z_entry, 3),
                "min_signal_strength": round(min_strength, 3),
                "max_position_size_pct": round(pos_size, 4),
                "ann_vol": round(ann_vol, 4),
                "hurst": round(hurst, 3),
                "dip_recovery": round(recovery_rate, 3),
                "beta": round(beta, 2),
            })
            
        for k in daily_params.keys():
            daily_params[k][i] = last_params[k]
            
    print(f"Time: {time.time() - t0:.3f}s")
    
    # Convert arrays to Series
    for k in daily_params.keys():
        daily_params[k] = pd.Series(daily_params[k], index=dates)
        
    return daily_params

if __name__ == "__main__":
    prices = pd.Series(np.random.randn(500).cumsum() + 100, index=pd.date_range("2020-01-01", periods=500))
    res = calibrate_expanding_profile("TEST", prices)
    print(res["z_score_entry_threshold"].tail())
