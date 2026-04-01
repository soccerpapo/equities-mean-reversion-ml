"""Per-stock adaptive parameter profiles.

Auto-calibrates strategy parameters (ATR stops, z-score thresholds,
position sizing) based on each stock's volatility, mean-reversion
strength, and dip recovery rate.  Supports manual overrides in settings.

Two volatility tiers:
  - Low-vol  (ann_vol <= 30%): tighter stops, looser entry, larger positions
  - High-vol (ann_vol > 30%):  wider stops, stricter entry, smaller positions

Mean-reversion strength (Hurst exponent) adjusts z-score threshold.
Dip recovery rate adjusts minimum signal strength.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StockProfile:
    """Per-stock parameter profile."""
    symbol: str
    # Calibrated parameters
    atr_stop_mult: float = 1.5
    atr_profit_mult: float = 2.5
    z_score_entry_threshold: float = 1.7
    min_signal_strength: float = 0.28
    max_position_size_pct: float = 0.12
    # Screener metrics used for calibration (stored for transparency)
    ann_vol: float = 0.25
    hurst: float = 0.5
    dip_recovery: float = 0.80
    beta: float = 1.0
    # Whether this profile was manually overridden
    is_override: bool = False


def calibrate_profile(
    symbol: str,
    prices: pd.Series,
    spy_returns: Optional[pd.Series] = None,
    base_atr_stop: float = 1.5,
    base_atr_profit: float = 2.5,
    base_z_entry: float = 1.7,
    base_min_strength: float = 0.28,
    base_max_pos: float = 0.12,
    update_freq: int = 21,
) -> StockProfile:
    """Calibrate a parameter profile from historical price data using an expanding window.
    To avoid look-ahead bias, it calibrates on data up to T-1 and applies to T.
    """
    from analysis.symbol_screener import _hurst_exponent, _dip_recovery_rate, _compute_beta

    returns = prices.pct_change().dropna()
    dates = returns.index
    n = len(dates)

    if n < 100:
        logger.warning(f"{symbol}: insufficient data for calibration, using defaults")
        return StockProfile(symbol=symbol)

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
        if i % update_freq == 0:
            window_prices = prices.iloc[:i+1]
            window_returns = returns.iloc[:i]

            ann_vol = float(window_returns.std() * np.sqrt(252))
            hurst = _hurst_exponent(window_returns)
            recovery_rate, _ = _dip_recovery_rate(window_prices, zscore_threshold=-base_z_entry)

            beta = 1.0
            if spy_returns is not None:
                beta = _compute_beta(window_returns, spy_returns.iloc[:i])

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

    profile = StockProfile(
        symbol=symbol,
        atr_stop_mult=pd.Series(daily_params["atr_stop_mult"], index=dates),
        atr_profit_mult=pd.Series(daily_params["atr_profit_mult"], index=dates),
        z_score_entry_threshold=pd.Series(daily_params["z_score_entry_threshold"], index=dates),
        min_signal_strength=pd.Series(daily_params["min_signal_strength"], index=dates),
        max_position_size_pct=pd.Series(daily_params["max_position_size_pct"], index=dates),
        ann_vol=pd.Series(daily_params["ann_vol"], index=dates),
        hurst=pd.Series(daily_params["hurst"], index=dates),
        dip_recovery=pd.Series(daily_params["dip_recovery"], index=dates),
        beta=pd.Series(daily_params["beta"], index=dates),
    )
    return profile


def calibrate_all(
    symbols: list,
    period: str = "5y",
    overrides: Optional[Dict[str, dict]] = None,
) -> Dict[str, StockProfile]:
    """Calibrate profiles for all symbols, applying manual overrides.

    Args:
        symbols: List of ticker symbols
        period: yfinance period for data fetch
        overrides: Optional dict of {symbol: {param: value}} for manual overrides

    Returns:
        Dict mapping symbol -> StockProfile
    """
    from data.fetcher import DataFetcher
    from config import settings

    fetcher = DataFetcher()
    overrides = overrides or getattr(settings, "STOCK_PROFILE_OVERRIDES", {})

    # Fetch SPY for beta calculation
    spy_df = fetcher.fetch_historical("SPY", period=period)
    spy_returns = spy_df["Close"].pct_change().dropna() if not spy_df.empty else None

    base_stop = getattr(settings, "ATR_STOP_MULTIPLIER", 1.5)
    base_profit = getattr(settings, "ATR_PROFIT_MULTIPLIER", 2.5)
    base_z = getattr(settings, "Z_SCORE_ENTRY_THRESHOLD", 1.7)
    base_str = getattr(settings, "MIN_SIGNAL_STRENGTH", 0.28)
    base_pos = getattr(settings, "MAX_POSITION_SIZE_PCT", 0.12)

    profiles = {}
    for symbol in symbols:
        if symbol == "SPY":
            continue

        df = fetcher.fetch_historical(symbol, period=period)
        if df.empty:
            logger.warning(f"No data for {symbol}, using defaults")
            profiles[symbol] = StockProfile(symbol=symbol)
            continue

        profile = calibrate_profile(
            symbol, df["Close"], spy_returns,
            base_atr_stop=base_stop,
            base_atr_profit=base_profit,
            base_z_entry=base_z,
            base_min_strength=base_str,
            base_max_pos=base_pos,
        )

        # Apply manual overrides
        if symbol in overrides:
            for param, value in overrides[symbol].items():
                if hasattr(profile, param):
                    setattr(profile, param, value)
                    profile.is_override = True

        profiles[symbol] = profile

    return profiles


def print_profiles(profiles: Dict[str, StockProfile]) -> None:
    """Print a formatted table of all stock profiles."""
    if not profiles:
        print("No profiles to display.")
        return

    print(f"\n{'='*95}")
    print("  PER-STOCK ADAPTIVE PROFILES (Latest Values)")
    print(f"{'='*95}")
    print(f"  {'Symbol':<8} {'ATR Stop':<10} {'ATR TP':<10} {'Z-Entry':<10} "
          f"{'MinStr':<10} {'MaxPos%':<10} {'AnnVol':<10} {'Hurst':<8} {'Recov':<8} {'Beta':<6}")
    print("-" * 95)

    from config import settings
    base_stop = getattr(settings, "ATR_STOP_MULTIPLIER", 1.5)
    base_profit = getattr(settings, "ATR_PROFIT_MULTIPLIER", 2.5)
    base_z = getattr(settings, "Z_SCORE_ENTRY_THRESHOLD", 1.7)
    base_str = getattr(settings, "MIN_SIGNAL_STRENGTH", 0.28)
    base_pos = getattr(settings, "MAX_POSITION_SIZE_PCT", 0.12)

    print(f"  {'GLOBAL':<8} {base_stop:<10.3f} {base_profit:<10.3f} {base_z:<10.3f} "
          f"{base_str:<10.3f} {base_pos:<10.2%} {'--':<10} {'--':<8} {'--':<8} {'--':<6}")
    print("-" * 95)

    for symbol, p in sorted(profiles.items()):
        override_flag = " *" if p.is_override else ""
        
        def get_val(val):
            return val.iloc[-1] if isinstance(val, pd.Series) else val
            
        atr_stop = get_val(p.atr_stop_mult)
        atr_profit = get_val(p.atr_profit_mult)
        z_entry = get_val(p.z_score_entry_threshold)
        min_str = get_val(p.min_signal_strength)
        pos_size = get_val(p.max_position_size_pct)
        ann_vol = get_val(p.ann_vol)
        hurst = get_val(p.hurst)
        recov = get_val(p.dip_recovery)
        beta = get_val(p.beta)

        print(f"  {symbol:<8} {atr_stop:<10.3f} {atr_profit:<10.3f} "
              f"{z_entry:<10.3f} {min_str:<10.3f} "
              f"{pos_size:<10.2%} {ann_vol:<10.1%} "
              f"{hurst:<8.3f} {recov:<8.0%} {beta:<6.2f}{override_flag}")

    print(f"\n  * = has manual overrides from STOCK_PROFILE_OVERRIDES")
    print(f"\n  Calibration logic:")
    print(f"    ATR stops:  wider for high-vol, tighter for low-vol (continuous scale)")
    print(f"    Z-entry:    stricter for trending stocks (high Hurst), looser for mean-reverting")
    print(f"    Position:   smaller for high-vol, larger for low-vol (risk parity)")
    print(f"    Min signal: looser for high dip-recovery, stricter for low recovery")
