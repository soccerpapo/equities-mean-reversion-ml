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
    bull_dip_recovery: float = 0.80
    bear_dip_recovery: float = 0.50
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
) -> StockProfile:
    """Calibrate a parameter profile from historical price data.

    Uses the same metrics as the screener but maps them directly to
    strategy parameters via continuous interpolation (not discrete tiers).

    Args:
        symbol: Ticker symbol
        prices: Close price Series (at least 252 bars)
        spy_returns: Optional SPY daily returns for beta calculation
        base_atr_stop: Global baseline ATR stop multiplier
        base_atr_profit: Global baseline ATR profit multiplier
        base_z_entry: Global baseline z-score entry threshold
        base_min_strength: Global baseline min signal strength
        base_max_pos: Global baseline max position size

    Returns:
        Calibrated StockProfile
    """
    from analysis.symbol_screener import (
        _hurst_exponent, _dip_recovery_rate, _dip_recovery_rate_by_regime, _compute_beta,
    )

    returns = prices.pct_change().dropna()
    if len(returns) < 100:
        logger.warning(f"{symbol}: insufficient data for calibration, using defaults")
        return StockProfile(symbol=symbol)

    ann_vol = float(returns.std() * np.sqrt(252))
    hurst = _hurst_exponent(returns)
    recovery_rate, _ = _dip_recovery_rate(prices, zscore_threshold=-base_z_entry)
    bull_recovery, bear_recovery, n_bull, n_bear = _dip_recovery_rate_by_regime(
        prices, zscore_threshold=-base_z_entry,
    )

    beta = 1.0
    if spy_returns is not None:
        beta = _compute_beta(returns, spy_returns)

    # --- ATR stop/profit multipliers ---
    # Continuous scale: at 15% vol -> 0.85x base, at 25% vol -> 1.0x, at 45% vol -> 1.5x
    vol_scale = np.clip(0.5 + (ann_vol - 0.15) / 0.30 * 1.0, 0.75, 1.6)
    atr_stop = base_atr_stop * vol_scale
    atr_profit = base_atr_profit * vol_scale

    # --- Z-score entry threshold ---
    # Lower Hurst (more mean-reverting) -> looser threshold (more signals)
    # H=0.35 -> z=1.4, H=0.50 -> z=1.7, H=0.65 -> z=2.1
    z_entry = base_z_entry + (hurst - 0.50) * 2.5
    z_entry = float(np.clip(z_entry, 1.3, 2.3))

    # --- Max position size ---
    # Inverse of volatility: low vol -> bigger positions, high vol -> smaller
    # 15% vol -> 15%, 25% vol -> 12%, 45% vol -> 7%
    pos_size = base_max_pos * (1.0 - (ann_vol - 0.25) * 1.5)
    pos_size = float(np.clip(pos_size, 0.05, 0.18))

    # --- Min signal strength ---
    # High recovery -> trust signals more (lower threshold)
    # 90% recovery -> 0.22, 80% -> 0.28, 65% -> 0.38
    strength_adjust = (recovery_rate - 0.80) * 0.6
    min_strength = base_min_strength - strength_adjust
    min_strength = float(np.clip(min_strength, 0.18, 0.45))

    profile = StockProfile(
        symbol=symbol,
        atr_stop_mult=round(atr_stop, 3),
        atr_profit_mult=round(atr_profit, 3),
        z_score_entry_threshold=round(z_entry, 3),
        min_signal_strength=round(min_strength, 3),
        max_position_size_pct=round(pos_size, 4),
        ann_vol=round(ann_vol, 4),
        hurst=round(hurst, 3),
        dip_recovery=round(recovery_rate, 3),
        bull_dip_recovery=round(bull_recovery, 3),
        bear_dip_recovery=round(bear_recovery, 3),
        beta=round(beta, 2),
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

    print(f"\n{'='*115}")
    print("  PER-STOCK ADAPTIVE PROFILES")
    print(f"{'='*115}")
    print(f"  {'Symbol':<8} {'ATR Stop':<10} {'ATR TP':<10} {'Z-Entry':<10} "
          f"{'MinStr':<10} {'MaxPos%':<10} {'AnnVol':<10} {'Hurst':<8} "
          f"{'BullRec':<8} {'BearRec':<8} {'Beta':<6}")
    print("-" * 115)

    from config import settings
    base_stop = getattr(settings, "ATR_STOP_MULTIPLIER", 1.5)
    base_profit = getattr(settings, "ATR_PROFIT_MULTIPLIER", 2.5)
    base_z = getattr(settings, "Z_SCORE_ENTRY_THRESHOLD", 1.7)
    base_str = getattr(settings, "MIN_SIGNAL_STRENGTH", 0.28)
    base_pos = getattr(settings, "MAX_POSITION_SIZE_PCT", 0.12)

    print(f"  {'GLOBAL':<8} {base_stop:<10.3f} {base_profit:<10.3f} {base_z:<10.3f} "
          f"{base_str:<10.3f} {base_pos:<10.2%} {'--':<10} {'--':<8} "
          f"{'--':<8} {'--':<8} {'--':<6}")
    print("-" * 115)

    for symbol, p in sorted(profiles.items()):
        override_flag = " *" if p.is_override else ""
        stop_delta = f"({p.atr_stop_mult - base_stop:+.2f})"
        z_delta = f"({p.z_score_entry_threshold - base_z:+.2f})"
        pos_delta = f"({(p.max_position_size_pct - base_pos) * 100:+.1f}pp)"

        print(f"  {symbol:<8} {p.atr_stop_mult:<10.3f} {p.atr_profit_mult:<10.3f} "
              f"{p.z_score_entry_threshold:<10.3f} {p.min_signal_strength:<10.3f} "
              f"{p.max_position_size_pct:<10.2%} {p.ann_vol:<10.1%} "
              f"{p.hurst:<8.3f} {p.bull_dip_recovery:<8.0%} "
              f"{p.bear_dip_recovery:<8.0%} {p.beta:<6.2f}{override_flag}")

    print(f"\n  * = has manual overrides from STOCK_PROFILE_OVERRIDES")
    print(f"\n  Calibration logic:")
    print(f"    ATR stops:  wider for high-vol, tighter for low-vol (continuous scale)")
    print(f"    Z-entry:    stricter for trending stocks (high Hurst), looser for mean-reverting")
    print(f"    Position:   smaller for high-vol, larger for low-vol (risk parity)")
    print(f"    Min signal: looser for high dip-recovery, stricter for low recovery")
