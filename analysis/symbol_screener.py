"""Symbol screener for identifying mean-reversion candidates.

Ranks candidate symbols by their suitability for the dip-buying strategy
using quantitative metrics: Hurst exponent, return autocorrelation,
variance ratio, dip recovery rate, liquidity, volatility, SPY correlation,
and beta.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """Estimate the Hurst exponent using the rescaled range (R/S) method.

    H < 0.5 => mean-reverting, H = 0.5 => random walk, H > 0.5 => trending.
    """
    ts = series.dropna().values
    if len(ts) < max_lag * 2:
        max_lag = len(ts) // 4

    lags = range(10, max(max_lag, 11))
    rs_values = []

    for lag in lags:
        n_chunks = len(ts) // lag
        if n_chunks < 1:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = ts[i * lag : (i + 1) * lag]
            mean_chunk = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_chunk)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_values.append((np.log(lag), np.log(np.mean(rs_chunk))))

    if len(rs_values) < 2:
        return 0.5

    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    slope, _ = np.polyfit(x, y, 1)
    return float(np.clip(slope, 0.0, 1.0))


def _variance_ratio(returns: pd.Series, holding_period: int = 5) -> float:
    """Compute the Lo-MacKinlay variance ratio.

    VR < 1 => mean-reverting, VR = 1 => random walk, VR > 1 => trending.
    """
    rets = returns.dropna().values
    n = len(rets)
    if n < holding_period * 2:
        return 1.0

    var_1 = np.var(rets, ddof=1)
    if var_1 == 0:
        return 1.0

    multi_rets = np.array([
        np.sum(rets[i : i + holding_period])
        for i in range(n - holding_period + 1)
    ])
    var_k = np.var(multi_rets, ddof=1)

    return float(var_k / (holding_period * var_1))


def _dip_recovery_rate(
    prices: pd.Series,
    zscore_threshold: float = -1.7,
    recovery_window: int = 10,
    zscore_window: int = 20,
) -> Tuple[float, int]:
    """Measure how often a z-score dip leads to positive forward returns.

    Returns:
        Tuple of (recovery_rate, number_of_dips)
    """
    rolling_mean = prices.rolling(window=zscore_window).mean()
    rolling_std = prices.rolling(window=zscore_window).std().replace(0, np.nan)
    zscore = (prices - rolling_mean) / rolling_std

    dip_mask = zscore < zscore_threshold
    dip_indices = prices.index[dip_mask]

    if len(dip_indices) == 0:
        return 0.0, 0

    recoveries = 0
    total_dips = 0
    for idx in dip_indices:
        loc = prices.index.get_loc(idx)
        if loc + recovery_window >= len(prices):
            continue
        total_dips += 1
        entry_price = prices.iloc[loc]
        future_max = prices.iloc[loc + 1 : loc + recovery_window + 1].max()
        if future_max > entry_price:
            recoveries += 1

    if total_dips == 0:
        return 0.0, 0
    return float(recoveries / total_dips), total_dips


def _dip_recovery_rate_by_regime(
    prices: pd.Series,
    zscore_threshold: float = -1.7,
    recovery_window: int = 10,
    zscore_window: int = 20,
    sma_period: int = 200,
) -> Tuple[float, float, int, int]:
    """Compute dip recovery rate separately for bull and bear regimes.

    Bull: price >= SMA(200) at the time of dip.
    Bear: price < SMA(200) at the time of dip.

    Returns:
        (bull_recovery_rate, bear_recovery_rate, n_bull_dips, n_bear_dips)
    """
    sma = prices.rolling(window=sma_period).mean()
    rolling_mean = prices.rolling(window=zscore_window).mean()
    rolling_std = prices.rolling(window=zscore_window).std().replace(0, np.nan)
    zscore = (prices - rolling_mean) / rolling_std

    dip_mask = zscore < zscore_threshold
    dip_indices = prices.index[dip_mask]

    bull_recoveries, bull_total = 0, 0
    bear_recoveries, bear_total = 0, 0

    for idx in dip_indices:
        loc = prices.index.get_loc(idx)
        if loc + recovery_window >= len(prices):
            continue
        if pd.isna(sma.iloc[loc]):
            continue

        entry_price = prices.iloc[loc]
        future_max = prices.iloc[loc + 1 : loc + recovery_window + 1].max()
        recovered = future_max > entry_price

        if prices.iloc[loc] >= sma.iloc[loc]:
            bull_total += 1
            if recovered:
                bull_recoveries += 1
        else:
            bear_total += 1
            if recovered:
                bear_recoveries += 1

    bull_rate = float(bull_recoveries / bull_total) if bull_total > 0 else 0.0
    bear_rate = float(bear_recoveries / bear_total) if bear_total > 0 else 0.0
    return bull_rate, bear_rate, bull_total, bear_total


def _compute_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Compute beta relative to benchmark."""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 30:
        return 1.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    var_bench = cov[1, 1]
    if var_bench == 0:
        return 1.0
    return float(cov[0, 1] / var_bench)


def screen_symbols(
    candidates: List[str],
    period: str = "5y",
    zscore_threshold: float = -1.7,
    verbose: bool = True,
) -> pd.DataFrame:
    """Screen candidate symbols for mean-reversion suitability.

    Fetches 5 years of data and computes:
      - Hurst exponent (want < 0.5)
      - Lag-1 return autocorrelation (want negative)
      - Variance ratio at 5d and 10d (want < 1.0)
      - Dip recovery rate (want high)
      - Avg daily dollar volume (want high)
      - Annualized volatility (want moderate: 15-40%)
      - SPY correlation (want 0.5-0.8)
      - Beta (want 0.8-1.3)
      - Composite score (weighted rank across all metrics)

    Args:
        candidates: List of ticker symbols to screen
        period: yfinance period string for data fetch
        zscore_threshold: z-score level to use for dip recovery test
        verbose: Print progress

    Returns:
        DataFrame with one row per symbol, sorted by composite score
    """
    from data.fetcher import DataFetcher

    fetcher = DataFetcher()

    # Fetch SPY as benchmark
    spy_df = fetcher.fetch_historical("SPY", period=period)
    if spy_df.empty:
        logger.error("Could not fetch SPY benchmark data")
        return pd.DataFrame()
    spy_returns = spy_df["Close"].pct_change().dropna()

    results = []
    for symbol in candidates:
        if verbose:
            print(f"  Screening {symbol}...", end=" ", flush=True)

        df = fetcher.fetch_historical(symbol, period=period)
        if df.empty:
            if verbose:
                print("SKIP (no data)")
            continue

        close = df["Close"]
        returns = close.pct_change().dropna()

        if len(returns) < 252:
            if verbose:
                print("SKIP (insufficient history)")
            continue

        # 1. Hurst exponent
        hurst = _hurst_exponent(returns)

        # 2. Lag-1 autocorrelation of daily returns
        autocorr_1 = float(returns.autocorr(lag=1))

        # 3. Variance ratios
        vr_5 = _variance_ratio(returns, holding_period=5)
        vr_10 = _variance_ratio(returns, holding_period=10)

        # 4. Dip recovery rate
        recovery_rate, n_dips = _dip_recovery_rate(
            close, zscore_threshold=zscore_threshold
        )

        # 5. Liquidity: avg daily dollar volume (last 60 days)
        recent = df.tail(60)
        avg_dollar_vol = float((recent["Close"] * recent["Volume"]).mean())

        # 6. Annualized volatility
        ann_vol = float(returns.std() * np.sqrt(252))

        # 7. SPY correlation (full period)
        aligned = pd.concat([returns, spy_returns], axis=1).dropna()
        spy_corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) > 30 else 0.0

        # 8. Beta
        beta = _compute_beta(returns, spy_returns)

        # 9. Signal frequency estimate: how many z-score dips per year
        rolling_mean = close.rolling(20).mean()
        rolling_std = close.rolling(20).std().replace(0, np.nan)
        zscore = (close - rolling_mean) / rolling_std
        n_years = len(close) / 252
        dips_per_year = (zscore < zscore_threshold).sum() / max(n_years, 0.5)

        row = {
            "symbol": symbol,
            "hurst": round(hurst, 3),
            "autocorr_1d": round(autocorr_1, 4),
            "var_ratio_5d": round(vr_5, 3),
            "var_ratio_10d": round(vr_10, 3),
            "dip_recovery": round(recovery_rate, 3),
            "n_dips": n_dips,
            "dips_per_year": round(dips_per_year, 1),
            "avg_dollar_vol_M": round(avg_dollar_vol / 1e6, 0),
            "ann_vol": round(ann_vol, 3),
            "spy_corr": round(spy_corr, 3),
            "beta": round(beta, 2),
        }
        results.append(row)
        if verbose:
            print(f"H={hurst:.3f}  AC={autocorr_1:+.4f}  VR5={vr_5:.3f}  "
                  f"Recov={recovery_rate:.0%}({n_dips})  "
                  f"Vol={ann_vol:.0%}  Beta={beta:.2f}")

    if not results:
        logger.warning("No symbols passed screening")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = _compute_composite_score(df_results)
    df_results = df_results.sort_values("composite_score", ascending=False)

    return df_results


def _compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a weighted composite score (0-100) based on all metrics.

    Scoring logic:
      - Hurst: lower is better (target < 0.5). Weight: 20%
      - Autocorr: more negative is better. Weight: 15%
      - Variance ratio 5d: lower is better (< 1.0). Weight: 10%
      - Dip recovery rate: higher is better. Weight: 20%
      - Liquidity: higher is better (log scale). Weight: 5%
      - Volatility: moderate is best (25% ideal, penalize extremes). Weight: 10%
      - SPY correlation: 0.65 ideal, penalize extremes. Weight: 10%
      - Beta: 1.05 ideal, penalize extremes. Weight: 10%
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        out["composite_score"] = []
        return out

    def _rank_pct(s: pd.Series, ascending: bool = True) -> pd.Series:
        """Rank as 0-1 percentile. ascending=True means lowest value gets highest score."""
        if ascending:
            return 1.0 - s.rank(pct=True, method="min")
        return s.rank(pct=True, method="min")

    # Hurst: lower is better
    s_hurst = _rank_pct(out["hurst"], ascending=True)

    # Autocorrelation: more negative is better
    s_autocorr = _rank_pct(out["autocorr_1d"], ascending=True)

    # Variance ratio 5d: lower is better
    s_vr = _rank_pct(out["var_ratio_5d"], ascending=True)

    # Dip recovery: higher is better
    s_recovery = _rank_pct(out["dip_recovery"], ascending=False)

    # Liquidity: higher is better (log scale)
    s_liq = _rank_pct(np.log1p(out["avg_dollar_vol_M"]), ascending=False)

    # Volatility: 25% is ideal, penalize distance from target
    vol_penalty = (out["ann_vol"] - 0.25).abs()
    s_vol = _rank_pct(vol_penalty, ascending=True)

    # SPY correlation: 0.65 is ideal
    corr_penalty = (out["spy_corr"] - 0.65).abs()
    s_corr = _rank_pct(corr_penalty, ascending=True)

    # Beta: 1.05 is ideal
    beta_penalty = (out["beta"] - 1.05).abs()
    s_beta = _rank_pct(beta_penalty, ascending=True)

    composite = (
        0.20 * s_hurst
        + 0.15 * s_autocorr
        + 0.10 * s_vr
        + 0.20 * s_recovery
        + 0.05 * s_liq
        + 0.10 * s_vol
        + 0.10 * s_corr
        + 0.10 * s_beta
    )

    out["composite_score"] = (composite * 100).round(1)
    return out


# Large-cap liquid candidates organized by sector for diversification
DEFAULT_CANDIDATES = [
    # Tech (already have AAPL, GOOGL, AMZN, NVDA, TSLA)
    "AMD", "AVGO", "CRM", "ADBE", "NFLX", "INTC", "QCOM", "ORCL",
    # Financials
    "JPM", "GS", "V", "MA", "BAC",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK",
    # Consumer
    "COST", "WMT", "HD", "MCD", "NKE", "SBUX",
    # Energy
    "XOM", "CVX",
    # Industrials
    "CAT", "BA", "UPS", "HON",
    # Communication
    "DIS", "CMCSA",
    # Previously dropped (include for comparison)
    "MSFT", "META",
]


def print_screen_report(df: pd.DataFrame) -> None:
    """Print a formatted screening report with interpretation."""

    if df.empty:
        print("No symbols to report.")
        return

    print(f"\n{'='*100}")
    print("  SYMBOL SCREENER: Mean-Reversion Suitability Ranking")
    print(f"{'='*100}")
    print(f"\n  Scoring: Hurst(20%) + AutoCorr(15%) + VarRatio(10%) + "
          f"DipRecovery(20%) + Liquidity(5%) + Vol(10%) + Corr(10%) + Beta(10%)")
    print(f"  Target profile: Hurst<0.5, AutoCorr<0, VR<1.0, "
          f"High recovery, Vol~25%, Corr~0.65, Beta~1.05")

    print(f"\n  {'Rank':<5} {'Symbol':<8} {'Score':<7} {'Hurst':<7} "
          f"{'AC(1d)':<9} {'VR(5d)':<8} {'VR(10d)':<8} "
          f"{'Recov%':<8} {'Dips':<6} {'Dip/Yr':<8} "
          f"{'$Vol(M)':<9} {'AnnVol':<8} {'Corr':<7} {'Beta':<6}")
    print("-" * 112)

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        hurst_flag = "*" if row["hurst"] < 0.45 else " "
        ac_flag = "*" if row["autocorr_1d"] < -0.02 else " "
        vr_flag = "*" if row["var_ratio_5d"] < 0.95 else " "
        rec_flag = "*" if row["dip_recovery"] >= 0.70 else " "

        print(f"  {rank:<5} {row['symbol']:<8} {row['composite_score']:<7.1f} "
              f"{row['hurst']:<6.3f}{hurst_flag} "
              f"{row['autocorr_1d']:+<8.4f}{ac_flag} "
              f"{row['var_ratio_5d']:<7.3f}{vr_flag} "
              f"{row['var_ratio_10d']:<7.3f}  "
              f"{row['dip_recovery']:<7.0%}{rec_flag} "
              f"{row['n_dips']:<6} {row['dips_per_year']:<7.1f}  "
              f"{row['avg_dollar_vol_M']:<8.0f}  "
              f"{row['ann_vol']:<7.1%}  "
              f"{row['spy_corr']:<6.3f} "
              f"{row['beta']:<6.2f}")

    print(f"\n  * = meets ideal threshold for that metric")

    # Interpretation
    top_n = min(10, len(df))
    top = df.head(top_n)
    print(f"\n  --- Top {top_n} Recommendations ---")
    for _, row in top.iterrows():
        strengths = []
        if row["hurst"] < 0.45:
            strengths.append("strong mean-reversion")
        if row["autocorr_1d"] < -0.02:
            strengths.append("negative autocorrelation")
        if row["dip_recovery"] >= 0.70:
            strengths.append(f"{row['dip_recovery']:.0%} dip recovery")
        if 0.15 <= row["ann_vol"] <= 0.35:
            strengths.append("moderate volatility")
        if 0.5 <= row["spy_corr"] <= 0.8:
            strengths.append("good SPY correlation")

        weakness = []
        if row["hurst"] >= 0.5:
            weakness.append("trending tendency")
        if row["dip_recovery"] < 0.50:
            weakness.append(f"low recovery ({row['dip_recovery']:.0%})")
        if row["ann_vol"] > 0.40:
            weakness.append("high volatility")
        if row["beta"] > 1.5:
            weakness.append(f"high beta ({row['beta']:.2f})")
        if row["n_dips"] < 5:
            weakness.append("few dip signals")

        s_str = ", ".join(strengths) if strengths else "none outstanding"
        w_str = ", ".join(weakness) if weakness else "none notable"
        print(f"  {row['symbol']:>6} (score {row['composite_score']:.1f}): "
              f"+ {s_str}  |  - {w_str}")

    # Sector diversification note
    print(f"\n  --- Existing Portfolio ---")
    print(f"  Current symbols: AAPL, GOOGL, AMZN, NVDA, TSLA (all tech/growth)")
    print(f"  Consider adding from different sectors for uncorrelated dip signals.")
