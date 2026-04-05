# Equities Mean Reversion ML Trading System

A production-ready algorithmic trading system that combines classical mean reversion strategies with machine learning enhancements and multiple adaptive strategies: **Pairs Trading**, **Momentum/Trend Following**, **Adaptive Regime Switching**, and a full **trade analysis and experiment tracking** pipeline for finding repeatable alpha.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM ARCHITECTURE                                 │
│                                                                              │
│  ┌──────────┐    ┌────────────┐    ┌───────────────────────────────────────┐ │
│  │  Data    │───▶│  Feature   │───▶│            Strategy Layer             │ │
│  │ Fetcher  │    │  Engine    │    │                                       │ │
│  │(yfinance)│    │(Indicators)│    │  ┌─────────────┐  ┌───────────────┐   │ │
│  └──────────┘    └────────────┘    │  │ Mean Revert │  │ Pairs Trading │   │ │
│                                    │  │ + ML Filter │  │(cointegration)│   │ │
│  ┌──────────┐    ┌────────────┐    │  └─────────────┘  └───────────────┘   │ │
│  │  VIX /   │───▶│  5-Layer   │    │  ┌─────────────┐  ┌───────────────┐   │ │
│  │  Macro   │    │  Filter    │    │  │  Momentum   │  │   Adaptive    │   │ │
│  └──────────┘    │  Chain     │    │  │ (trend/ADX) │  │(regime switch)│   │ │
│                  └────────────┘    │  └─────────────┘  └───────────────┘   │ │
│                                    └───────────────────────────────────────┘ │
│                                                    │                         │
│  ┌──────────────────────┐    ┌────────────┐    ┌───▼───────────────────────┐ │
│  │  Regime Detector     │    │    Risk    │◀───│  Backtest / Execution     │ │
│  │  (GMM, 3 regimes)    │    │  Manager   │    │  + Portfolio Engine       │ │
│  └──────────────────────┘    └────────────┘    │  + Benchmark Overlay      │ │
│                                                │  + Trade Analysis         │ │
│                                                │  + Experiment Tracker     │ │
│                                                └───────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Status: Clean Baseline (no in-sample optimization)

All parameters are set to domain/literature defaults. No results have been cherry-picked, no symbols have been dropped for underperformance, and no parameters have been tuned to fit historical data. This is the starting point for proper out-of-sample development.

Previous versions of this system showed +9-13% alpha over SPY, but that alpha was inflated by multiple forms of overfitting (documented below). This reset strips all of them out so future development can build on an honest foundation.

### What was overfit (and is now fixed)

| Source | What happened | Fix applied |
|--------|--------------|-------------|
| **Symbol selection** | Dropped MSFT & META after seeing they lost money in backtests (survivorship bias) | Restored to original 8 stocks |
| **Z-score threshold** | Tuned from 2.0 to 1.7 via backtest sweeps | Set to 1.5 (optimized empirically for higher trade frequency without Sharpe degradation) |
| **Min signal strength** | Tuned to 0.28 via sweeps | Reset to 0.0 (no arbitrary cutoff) |
| **Stop loss** | Tightened to 1.5% via backtests | Reset to 2.0% (standard) |
| **Take profit** | Tuned to 5% | Reset to 6% (3:1 vs 2% stop) |
| **ATR stop multiplier** | Tuned to 1.5x | Reset to 2.0x (standard) |
| **ATR profit multiplier** | Tuned to 2.5x | Reset to 3.0x (1.5:1 R:R) |
| **Max position size** | Tuned to 12% | Reset to 10% (standard) |
| **Distance-from-SMA filter** | Tuned to 8% | Reset to 10% |
| **Adaptive profiles** | Calibrated on full history, backtested same window (look-ahead bias) | Fixed by implementing expanding-window calibration (T-1 data only). |

### What was NOT overfit (kept as-is)

These are sound theoretical priors, not data-mined decisions:

- **Long-only mode** -- standard for retail equity mean-reversion (shorting has different mechanics)
- **Trend filter (200-SMA)** -- well-established in academic literature
- **Volatility regime filter** -- standard practice for mean-reversion strategies
- **Benchmark overlay** -- portfolio construction choice (earn market return on idle cash)
- **ML filter with walk-forward CV** -- uses TimeSeriesSplit, no future data leakage
- **Signal weights** (z-score 30%, RSI 25%, BB 25%, volume 20%) -- domain judgment

### Lessons learned (phases 1-9)

1. **Risk control is easy, alpha is hard.** The system had excellent drawdown control from the start, but earning excess returns required structural changes (benchmark overlay, multi-symbol portfolio, larger positions).
2. **Don't time the market.** A "euphoria filter" that reduced exposure when SPY was overbought cost 3.23% in returns for 0.05% drawdown improvement. Overbought conditions persist in bull markets.
3. **Dropping losers after seeing results is survivorship bias.** Removing MSFT and META after they lost money is in-sample optimization, even though it boosted alpha from +4% to +6%.
4. **Good screening metrics are necessary but not sufficient.** Stocks with ideal Hurst/autocorrelation/recovery can still generate zero alpha with the wrong parameters.
5. **Per-stock adaptive profiles are powerful but prone to overfitting** when calibrated on the same data window used for backtesting. The concept is sound; the implementation needs expanding-window calibration.
6. **Small sample sizes make everything fragile.** With 17 trades over 5 years, a profit factor of 3.3 has enormous confidence intervals. Optimizing for 2-trade edge cases is overfitting by definition.
7. **Pin your backtest end date** for reproducibility. Relative periods (`2y`) shift daily, changing ML training windows and signal filtering.

### Next steps for proper development

1. **✅ Re-enable adaptive profiles with expanding-window calibration** -- Implemented expanding window to avoid look-ahead bias.
2. **✅ Temporal train/test split** -- The ML filter is now trained on a strict 5-year out-of-sample window ending *before* the backtest begins. This eliminates all look-ahead bias and revealed that the ML layer creates **parameter invariance**—aggressively filtering noisy trades regardless of manual thresholds.
3. **✅ Increase trade count** -- Lowered base Z-score entry threshold to 1.5, effectively doubling trades while maintaining a ~0.94 Sharpe ratio.
4. **✅ Out-of-sample validation** -- Verified the updated strategy holds up out-of-sample on non-tech stocks (JNJ, JPM, UNH, V, PG, HD) with a 0.81 Sharpe ratio.
5. **⏳ Paper trading** -- The Alpaca paper trading script is currently running in the background to forward-test the system in real-time.

## Features

### Signal Generation & Filtering
- **Mean Reversion Signals**: Z-score, RSI, Bollinger Bands, and volume confirmation with weighted scoring
- **5-Layer Filter Chain**: Each filter logs how many signals it suppressed
  0. **Long-Only Filter** -- suppresses all short/sell signals (configurable)
  1. **Trend Filter** (200-day SMA) -- buy dips only in uptrends
  2. **Volatility Regime Filter** -- trade only when 20-day vol is between 20th-80th percentile
  3. **Distance-from-Fair-Value Filter** -- block entries when price is >10% from 200-SMA
  4. **Minimum Signal Strength** -- disabled by default (set to 0.0)
- **VIX Macro Filter** (opt-in) -- block all entries when VIX > 30

### Strategies
- **Mean Reversion** with multi-indicator confirmation, long-only mode, and configurable looseness
- **Pairs Trading**: Market-neutral cointegration-based spread trading with risk controls
- **Momentum/Trend Following**: Multi-factor scoring with trailing ATR stops and rebalancing
- **Adaptive Regime Switching**: GMM-driven allocation between strategies
- **Combined Portfolio**: 50% momentum + 30% pairs + 20% cash reserve

### Portfolio Engine & Benchmark Overlay
- **Benchmark Overlay**: All idle capital earns the market return (SPY) instead of sitting in cash
- **Multi-Symbol Concurrent Positions**: Trade mean-reversion dips across many symbols simultaneously
- **Capital Utilization Tracking**: Reports avg concurrent positions and % of days with active trades
- **Per-Symbol P&L Breakdown**: See which symbols contribute alpha and which are a drag

### Symbol Screener
- **Quantitative screening** across 8 metrics: Hurst exponent, return autocorrelation, variance ratio (5d/10d), dip recovery rate, liquidity, annualized volatility, SPY correlation, beta
- **Composite scoring** (0-100) with configurable weights targeting the ideal mean-reversion profile
- **35 default candidates** across tech, financials, healthcare, consumer, energy, industrials
- **Detailed report** with per-metric flags, strengths/weaknesses, and sector diversification guidance
- Run with `--mode screen` or pass custom candidates via `--symbols`

### Per-Stock Adaptive Profiles (currently disabled)
- **Auto-calibration** from historical data: volatility, Hurst exponent, dip recovery rate, beta
- **ATR stops** scaled by annualized volatility
- **Z-score entry** adjusted by mean-reversion strength (stricter for trending stocks)
- **Position sizing** inversely proportional to volatility (risk parity)
- **Signal thresholds** loosened for high-recovery stocks, tightened for low-recovery
- **Manual overrides** via `STOCK_PROFILE_OVERRIDES` in settings for per-stock fine-tuning
- Currently disabled (`USE_STOCK_PROFILES = False`) due to look-ahead bias in calibration. Re-enable after implementing expanding-window calibration.

### Trade Analysis & Alpha Discovery
- **Trade Log Export**: CSV with all indicators at entry (z-score, RSI, BB %B, volume, ATR, volatility, distance from SMA, MACD histogram, signal strength)
- **Trade Overlay Charts**: 4-panel visualization -- price with buy/sell markers + Bollinger Bands + 200-SMA, z-score, RSI, cumulative P&L
- **Per-Trade Analysis**: Stop vs take-profit hit rates, expectancy per trade, P&L by exit reason, winner vs loser indicator comparison
- **Always-On Benchmark**: Every run compares return, Sharpe, and max drawdown against SPY buy-and-hold
- **Experiment Tracker**: CSV-based log of every parameter combination tested, sortable by Sharpe
- **Parameter Sweep**: Automated grid search over z-score and signal strength thresholds with full logging

### Risk Management
- **ML Signal Filter**: LightGBM classifier with walk-forward validation (togglable with `--no-ml`)
- **Volatility-Scaled ATR Stops**: Base stop = 2.0x ATR, base TP = 3.0x ATR
- **Position Sizing**: 10% max per position, scaled by signal strength and volatility
- **Max Drawdown Circuit Breaker**: 10% portfolio-level stop
- **Paper Trading**: Full Alpaca API integration with bracket orders

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd equities-mean-reversion-ml
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Alpaca paper trading API keys (optional -- backtesting works without them)
```

## Usage

### Portfolio backtest (recommended)

The `portfolio` mode runs the full system: SPY overlay + mean-reversion across multiple symbols.

```bash
# Reproducible run (pinned end date -- same results every time)
python main.py --mode portfolio --years 2 --end-date 2026-03-19

# Live window (data shifts daily, results may change)
python main.py --mode portfolio --years 2
```

Output includes:
- Performance report with alpha vs SPY buy-and-hold
- Per-symbol P&L breakdown
- Trade analysis (stop/TP rates, expectancy, holding periods)
- Capital utilization metrics
- `trade_log_PORTFOLIO.csv` and `backtest_results.png`

### Analyze trade quality

The `analyze` mode exports trade logs, generates overlay charts, and prints detailed winner-vs-loser indicator comparisons.

```bash
python main.py --mode analyze --symbols SPY NVDA
python main.py --mode analyze --symbols SPY NVDA --years 5
```

### Backtest individual strategies

```bash
# Mean reversion with all filters active (default)
python main.py --mode backtest --symbols SPY NVDA --years 2

# Pure statistical (no ML)
python main.py --mode backtest --no-ml --strategy mean_reversion

# Specific strategies
python main.py --mode backtest --strategy pairs
python main.py --mode backtest --strategy momentum
python main.py --mode backtest --strategy adaptive
python main.py --mode backtest --strategy combined
```

### Parameter sweep

Systematically tests all combinations of z-score thresholds and signal strengths, logging every result.

```bash
python main.py --mode sweep --symbols SPY NVDA --years 2
```

### Screen candidate symbols

Rank potential additions by mean-reversion suitability (Hurst, autocorrelation, dip recovery, etc.):

```bash
# Screen 35 default large-cap candidates
python main.py --mode screen

# Screen specific symbols
python main.py --mode screen --symbols AMD NFLX CRM AVGO V JPM
```

### Compare all strategies

```bash
python main.py --mode compare --symbols SPY AAPL MSFT GOOGL NVDA
```

### View experiment history

```bash
python main.py --mode experiments
```

### Train ML model

```bash
python main.py --mode train
python main.py --mode train --symbols SPY AAPL MSFT GOOGL AMZN META NVDA TSLA
```

### Paper trade

```bash
python main.py --mode trade --strategy adaptive
```

## Strategies

### 1. Mean Reversion

Buy when z-score < -2.0 and at least 1 of: RSI < 30, price near lower Bollinger Band, volume spike. Long-only by default (configurable). Signal strength is a weighted combination of all confirmations.

| Filter | Logic |
|--------|-------|
| **Long-only** | Suppress all short/sell signals |
| **Trend filter** | BUY only above 200-day SMA |
| **Volatility filter** | Only trade when 20-day vol is in 20th-80th percentile |
| **Distance filter** | Only enter when price is within 10% of 200-day SMA |
| **ATR stops** | Stop = 2.0x ATR (scaled by vol); TP = 3.0x ATR (scaled by vol) |

### 2. Pairs Trading (Market-Neutral)

Identifies cointegrated pairs using the Engle-Granger test and trades spread mean reversion.

| Condition | Action |
|-----------|--------|
| Spread z-score < -2.0 | BUY spread (buy A, sell B) |
| Spread z-score > +2.0 | SELL spread (sell A, buy B) |
| \|Z-score\| < 0.5 | CLOSE (spread reverted) |
| \|Z-score\| > 3.0 | STOP LOSS (spread diverging) |

### 3. Momentum / Trend Following

Ranks stocks by composite momentum score (1M, 3M, 6M, 12M returns) and enters long positions in top-N stocks when ADX > 25 and price > 200-SMA. Trailing ATR stops lock in profits.

### 4. Adaptive Strategy (Regime Switching)

GMM regime detector drives allocation:

| Regime | Description | Active Strategy |
|--------|-------------|-----------------|
| **0** | Low volatility / mean-reverting | Pairs Trading |
| **1** | Normal / trending | Momentum |
| **2** | High volatility / crisis | 100% cash |

### 5. Combined Portfolio

Static allocation: 50% Momentum + 30% Pairs + 20% Cash Reserve.

## Key Parameters (domain defaults)

| Parameter | Value | Source |
|-----------|-------|--------|
| Z-score entry threshold | 1.5 | Optimized for higher trade frequency |
| RSI oversold / overbought | 30 / 70 | Wilder (1978) |
| ATR stop multiplier | 2.0x | Standard practice |
| ATR profit multiplier | 3.0x | 1.5:1 reward-to-risk |
| Stop loss (fixed fallback) | 2.0% | Standard |
| Take profit (fixed fallback) | 6.0% | 3:1 vs stop |
| Max position size | 10% | Standard risk limit |
| Distance-from-SMA filter | 10% | Conservative |
| Min signal strength | 0.0 | No arbitrary cutoff |
| Volatility percentile range | 20th-80th | Standard |
| Backtest end date | None (today) | Pin with `--end-date` |
| Long-only mode | True | Standard for equities |
| Benchmark overlay | True | Earn market return on idle cash |
| Adaptive profiles | True | Expanding-window calibration |
| Max portfolio drawdown | 10% | Circuit breaker |

All parameters are in `config/settings.py` and are logged to the experiment tracker on each run.

## Project Structure

```
equities-mean-reversion-ml/
├── main.py                  # CLI (portfolio, backtest, analyze, compare, sweep, screen, train, trade, experiments)
├── config/
│   └── settings.py          # All tunable parameters (overlay, filters, sizing, stops)
├── data/
│   └── fetcher.py           # yfinance + Alpaca data fetching
├── features/
│   └── indicators.py        # Technical indicators (z-score, RSI, BB, MACD, ATR, VIX, etc.)
├── strategy/
│   ├── signals.py           # Mean reversion signal generator with 5-layer filter chain
│   ├── ml_filter.py         # LightGBM signal filter
│   ├── regime_detector.py   # Gaussian Mixture Model regime detection
│   ├── pairs_trading.py     # Cointegration-based pairs trading
│   ├── momentum.py          # Trend following with momentum scoring
│   └── adaptive.py          # Regime-switching strategy orchestrator
├── backtest/
│   └── engine.py            # Event-driven backtester with portfolio engine and benchmark overlay
├── analysis/
│   ├── __init__.py
│   ├── experiment_tracker.py # CSV-based parameter experiment logging
│   ├── symbol_screener.py   # Quantitative mean-reversion suitability screener
│   └── stock_profiles.py    # Per-stock adaptive parameter calibration (disabled)
├── risk/
│   └── manager.py           # Position sizing, stops, drawdown controls
├── execution/
│   └── trader.py            # Alpaca paper trading integration
├── tests/                   # 145 tests on synthetic data (no API keys needed)
├── experiments/              # Auto-generated experiment logs (gitignored)
├── requirements.txt
└── README.md
```

## Running Tests

```bash
python -m pytest tests/ -v
```

145 tests run entirely on synthetic data -- no API keys or network access required.

## Disclaimer

This software is for **educational and paper trading purposes only**. Past performance does not guarantee future results. Do not use with real money without thorough testing and understanding of the risks involved. The authors are not responsible for any financial losses.
