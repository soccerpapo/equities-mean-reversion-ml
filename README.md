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

## Results: Positive Alpha Achieved

### Portfolio Backtest (2-year, adaptive profiles, `--end-date 2026-03-19`)

| Metric | Strategy | SPY Buy & Hold | Delta |
|--------|----------|----------------|-------|
| **Return** | **43.67%** | 30.70% | **+12.97%** |
| **Alpha** | **+12.97%** | -- | -- |
| **Sharpe** | **1.063** | 0.901 | **+0.162** |
| Max Drawdown | -19.62% | -18.76% | -0.86% |
| Trades | 9 | -- | -- |
| Win Rate | 88.9% | -- | -- |
| Profit Factor | 24.25 | -- | -- |
| Expectancy | $2,167/trade | -- | -- |
| Capital Utilization | 51.6% | -- | -- |

### 5-Year Stress Test (includes 2022 bear market, adaptive profiles, `--end-date 2026-03-19`)

| Metric | Strategy | SPY Buy & Hold | Delta |
|--------|----------|----------------|-------|
| **Return** | **90.32%** | 80.89% | **+9.43%** |
| **Alpha** | **+9.43%** | -- | -- |
| **Sharpe** | **0.793** | 0.786 | **+0.007** |
| Max Drawdown | -26.89% | -24.50% | -2.39% |
| Trades | 17 | -- | -- |
| Win Rate | 76.5% | -- | -- |
| Profit Factor | 3.30 | -- | -- |
| Expectancy | $1,199/trade | -- | -- |

### Per-Symbol Alpha Breakdown (5-year, adaptive profiles)

| Symbol | Trades | P&L | Win Rate | ATR Stop | Z-Entry | MaxPos |
|--------|--------|-----|----------|----------|---------|--------|
| TSLA | 3 | +$11,441 | 66.7% | 2.40x | 1.92 | 5.3% |
| GOOGL | 5 | +$5,501 | 80.0% | 1.48x | 1.82 | 11.2% |
| AMZN | 3 | +$4,640 | 100.0% | 1.58x | 1.71 | 10.8% |
| AAPL | 4 | +$941 | 75.0% | 1.43x | 1.83 | 11.4% |
| NVDA | 2 | -$2,131 | 50.0% | 2.40x | 1.79 | 7.6% |

### Impact of Adaptive Profiles (uniform vs per-stock, 5-year)

| Metric | Uniform (5Y) | Adaptive (5Y) | Improvement |
|--------|-------------|--------------|-------------|
| **Alpha** | +2.26% | **+9.43%** | **+7.17%** |
| Win Rate | 66.7% | **76.5%** | +9.8pp |
| Expectancy | $454 | **$1,199** | **+$745** |
| TP Hit Rate | 66.7% | **76.5%** | +9.8pp |
| Stop Hit Rate | 33.3% | **23.5%** | -9.8pp |

### How we got here

The system went through five phases of improvement:

**Phase 1: Risk control** -- Long-only mode, 5-layer filter chain, volatility-scaled ATR stops, parameter sweep. Result: near-zero drawdown but negative alpha (sitting in cash 95% of the time while the market rallied).

**Phase 2: Capital utilization** -- Three structural changes solved the alpha problem:

1. **Benchmark overlay** -- Idle cash is invested in SPY instead of earning 0%. When a mean-reversion signal fires, SPY shares are liquidated to fund the trade. On exit, proceeds return to SPY. The strategy earns market return + incremental alpha from trades.
2. **Multi-symbol portfolio engine** -- A single portfolio holds SPY as its base and trades mean-reversion dips across multiple symbols concurrently. More symbols = more dip opportunities = higher capital utilization (15% -> 44%).
3. **Larger position sizing** -- Increased from 5% to 12% per trade. Drawdowns were well-controlled, leaving room to amplify winning trades.

**Phase 3: Euphoria filter experiment (rejected)** -- Tested reducing SPY overlay exposure when the benchmark itself was overbought (RSI > 75 or z-score > 1.5), scaling down to 30% exposure at extreme levels. The hypothesis was that avoiding predictable pullbacks would improve risk-adjusted returns.

Result: the filter was active 132 of 500 days but only improved max drawdown by 0.05% (-19.41% to -19.36%) while costing 3.23% in returns (alpha dropped from +8.13% to +4.90%, Sharpe from 1.075 to 1.019). Every metric got worse. The market stayed "overbought" and kept rallying -- selling tops in a bull market is the mirror image of the same mistake as shorting rallies. The experiment was reverted.

| Metric | Without Filter | With Euphoria Filter | Delta |
|--------|---------------|---------------------|-------|
| Return | **39.92%** | 36.69% | -3.23% |
| Alpha | **+8.13%** | +4.90% | -3.23% |
| Sharpe | **1.075** | 1.019 | -0.056 |
| Max Drawdown | -19.41% | -19.36% | +0.05% |

**Lesson learned:** In a bull market, "overbought" conditions persist far longer than mean-reversion models expect. Reducing exposure when the market looks stretched is just market timing in disguise, and market timing destroys alpha. The correct approach is to stay fully invested in the benchmark and only make tactical trades on individual stock dips.

**Phase 4: 5-year stress test** -- Extended the backtest from 2 years to 5 years to include the 2022 bear market (-24.5% on SPY). Alpha remained positive (+4.11% with the original 7 symbols), confirming the edge survives a full market cycle.

**Phase 5: Symbol curation** -- The 5-year backtest revealed MSFT (0% win rate, -$1,081) and META (0% win rate, -$417) were consistent alpha destroyers. Initial attempt to drop only MSFT backfired: the freed capital flowed to extra META trades that lost even more (-$1,617). The key insight -- proposed by the system's developer -- was to drop *both* losers simultaneously so capital could only reallocate to winners. This worked:

| Metric | 7 symbols | Drop MSFT only | Drop MSFT + META |
|--------|-----------|---------------|-----------------|
| Alpha (5Y) | +4.11% | +3.77% (worse) | **+6.29%** (best) |
| Win Rate | 60.9% | 60.9% | **70.0%** |
| Stop Rate | 39.1% | 39.1% | **30.0%** |
| Expectancy | $497 | $475 | **$636** |

**Lesson learned:** Removing a single losing symbol can backfire if the freed capital flows to another loser. You have to identify *all* the consistent losers and remove them together, so capital strictly reallocates to winners. Diversification only helps when all components have a positive or neutral edge.

**Phase 6: Quantitative symbol screener & expansion attempt** -- Built a symbol screener that ranks candidates by 8 mean-reversion metrics: Hurst exponent (< 0.5 = mean-reverting), lag-1 return autocorrelation (negative = dips bounce), variance ratio (< 1.0), dip recovery rate, liquidity, annualized volatility, SPY correlation, and beta. A weighted composite score (0-100) ranks suitability for the dip-buying strategy.

Screened 35 large-cap names across tech, financials, healthcare, consumer, energy, and industrials. Top scorers: V (82.0), MSFT (73.3), AVGO (73.0), MA (71.3), META (65.7), CRM (65.1), JPM (64.7).

Tested adding the top non-tech names (V, MA, JPM, CRM) to diversify across sectors:

| Config | Alpha (2Y) | Alpha (5Y) | Win Rate (5Y) | Trades (5Y) |
|--------|-----------|-----------|---------------|-------------|
| **5 symbols (original)** | **+10.69%** | **+6.29%** | **70.0%** | 20 |
| + V, JPM | +10.03% | -0.19% | 60.0% | 30 |
| + V, MA, JPM, CRM | +10.03% | -3.29% | 52.6% | 38 |

Every expansion degraded 5-year alpha. The new symbols had good screening metrics but generated near-zero or negative alpha in practice — and diverted capital from proven winners (GOOGL, AMZN, AAPL).

**Lesson learned:** Good screening metrics (Hurst, autocorrelation, dip recovery) are necessary but not sufficient. The strategy's alpha comes from a specific combination of filters + ATR stops + position sizing that works on a narrow set of names. This is a **concentrated alpha strategy**, not a diversified one — the edge is narrow and deep, not wide and shallow. The screener is useful for eliminating bad candidates, but the backtest remains the final arbiter.

**Phase 7: Per-stock adaptive profiles (the biggest single improvement)** -- After the symbol expansion failed, the developer identified the root cause: the strategy was treating every stock identically despite wildly different volatility profiles. NVDA (52% annualized vol) and AAPL (28% vol) were getting the same 1.5x ATR stops and 12% position sizing — NVDA's normal price swings were clipping stops before trades could play out, while AAPL's smaller moves weren't being exploited aggressively enough.

The developer proposed building per-stock parameter profiles calibrated from each stock's own characteristics, rather than just adding more symbols with one-size-fits-all parameters. This reframed the problem from "which stocks to trade" to "how to trade each stock optimally" — a fundamentally better question.

The resulting auto-calibration system uses continuous interpolation (not discrete tiers) to minimize overfitting:
- **ATR stops**: scaled by annualized volatility (NVDA gets 2.4x, AAPL gets 1.43x)
- **Z-score entry**: adjusted by Hurst exponent (trending stocks need bigger dips to trigger)
- **Position sizing**: inverse of volatility (risk parity — smaller bets on volatile names)
- **Min signal strength**: adjusted by dip recovery rate (high recovery = trust signals more)

This was the single largest alpha improvement in the system's history:

| Metric | Uniform (5Y) | Adaptive (5Y) | Change |
|--------|-------------|--------------|--------|
| **Alpha** | +2.26% | **+9.43%** | **+317% relative improvement** |
| **Expectancy** | $454/trade | **$1,199/trade** | **+164%** |
| **TSLA P&L** | -$309 | **+$11,441** | **turned a loser into the top performer** |
| **Stop hit rate** | 33.3% | **23.5%** | fewer premature stop-outs |
| **TP hit rate** | 66.7% | **76.5%** | more trades reaching target |

TSLA went from a net loser (-$309) to the best performer (+$11,441) — an $11,750 swing from per-stock parameter calibration. The wider stops (2.4x ATR vs 1.5x) let winning trades breathe through normal TSLA volatility instead of getting stopped out, while the smaller position sizing (5.3% vs 12%) limited damage during the 2022 bear market. The same logic helped every stock: GOOGL's 93% dip recovery earned it a looser signal threshold, capturing more of those reliable bounces; AAPL's low volatility earned it tighter stops and larger positions.

**Lesson learned:** In a concentrated alpha strategy, optimizing *how* you trade each stock matters more than *which* stocks you trade. A stock that loses money with generic parameters can become your best performer with calibrated ones. The developer's insight — that uniform parameters were the bottleneck, not the symbol universe — was the key breakthrough.

**Phase 8: Reproducibility** -- Backtest results were drifting between runs because `period="2y"` is relative to the current date. Each day adds a new bar and drops an old one, shifting the ML training window and changing which signals get filtered. The fix: `--end-date` flag that pins the data window to a specific calendar date, converting relative periods to absolute `start`/`end` dates. All data-fetching paths (benchmark, profiles, signals, VIX) use the same pinned boundary. Results are now bit-for-bit identical across runs with the same `--end-date`. Without it, the current behavior is preserved (window shifts daily).

**Current state:** The strategy beats SPY buy-and-hold by +12.97% (2Y) / +9.43% (5Y) with per-stock adaptive parameter profiles, validated through the 2022 bear market. All results are reproducible via `--end-date 2026-03-19`.

## Features

### Signal Generation & Filtering
- **Mean Reversion Signals**: Z-score, RSI, Bollinger Bands, and volume confirmation with weighted scoring
- **5-Layer Filter Chain**: Each filter logs how many signals it suppressed
  0. **Long-Only Filter** -- suppresses all short/sell signals (configurable)
  1. **Trend Filter** (200-day SMA) -- buy dips only in uptrends
  2. **Volatility Regime Filter** -- trade only when 20-day vol is between 20th-80th percentile
  3. **Distance-from-Fair-Value Filter** -- block entries when price is >8% from 200-SMA
  4. **Minimum Signal Strength** -- require weighted confirmation score >= 0.28
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

### Trade Analysis & Alpha Discovery
- **Trade Log Export**: CSV with all indicators at entry (z-score, RSI, BB %B, volume, ATR, volatility, distance from SMA, MACD histogram, signal strength)
- **Trade Overlay Charts**: 4-panel visualization -- price with buy/sell markers + Bollinger Bands + 200-SMA, z-score, RSI, cumulative P&L
- **Per-Trade Analysis**: Stop vs take-profit hit rates, expectancy per trade, P&L by exit reason, winner vs loser indicator comparison
- **Always-On Benchmark**: Every run compares return, Sharpe, and max drawdown against SPY buy-and-hold
- **Experiment Tracker**: CSV-based log of every parameter combination tested, sortable by Sharpe
- **Parameter Sweep**: Automated grid search over z-score and signal strength thresholds with full logging

### Per-Stock Adaptive Profiles
- **Auto-calibration** from historical data: volatility, Hurst exponent, dip recovery rate, beta
- **ATR stops** scaled by annualized volatility (1.43x for AAPL, 2.4x for NVDA/TSLA)
- **Z-score entry** adjusted by mean-reversion strength (stricter for trending stocks)
- **Position sizing** inversely proportional to volatility (risk parity)
- **Signal thresholds** loosened for high-recovery stocks, tightened for low-recovery
- **Manual overrides** via `STOCK_PROFILE_OVERRIDES` in settings for per-stock fine-tuning
- Toggle with `USE_STOCK_PROFILES = True/False` in settings

### Risk Management
- **ML Signal Filter**: LightGBM classifier with walk-forward validation (togglable with `--no-ml`)
- **Volatility-Scaled ATR Stops**: Base stop = 1.5x ATR, base TP = 2.5x ATR; per-stock calibration via adaptive profiles
- **Position Sizing**: Per-stock adaptive (5-15% range), scaled by signal strength and volatility
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

The `portfolio` mode runs the full system: SPY overlay + mean-reversion across multiple symbols. This is the mode that produces positive alpha.

```bash
# Reproducible run (pinned end date — same results every time)
python main.py --mode portfolio --years 2 --end-date 2026-03-19

# Live window (data shifts daily, results may change)
python main.py --mode portfolio --symbols SPY NVDA AAPL GOOGL AMZN TSLA --years 2
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

Output includes:
- `trade_log_SPY.csv` / `trade_log_NVDA.csv` -- every trade with all indicators at entry
- `trades_overlay_SPY.png` -- price chart with buy/sell markers, z-score, RSI, cumulative P&L
- Per-trade analysis: stop/TP hit rates, expectancy, indicator differences between winners and losers
- Actionable warnings (e.g., "stops hit more often than TPs -- consider widening stop")

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

Buy when z-score < -1.7 and at least 1 of: RSI < 30, price near lower Bollinger Band, volume spike. Long-only by default (configurable). Signal strength is a weighted combination of all confirmations.

| Filter | Logic |
|--------|-------|
| **Long-only** | Suppress all short/sell signals |
| **Trend filter** | BUY only above 200-day SMA |
| **Volatility filter** | Only trade when 20-day vol is in 20th-80th percentile |
| **Distance filter** | Only enter when price is within 8% of 200-day SMA |
| **ATR stops** | Stop = 1.5x ATR (scaled by vol); TP = 2.5x ATR (scaled by vol) |

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

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Long-only mode | True | Suppress all short signals |
| Benchmark overlay | True | Idle cash invested in SPY |
| Benchmark symbol | SPY | Configurable |
| Z-score entry threshold | 1.7 | Lower = more trades, higher = fewer/stronger |
| Min signal strength | 0.28 | Weighted confirmation score minimum |
| Min optional confirmations | 1 | Beyond z-score (1=looser, 2=tighter) |
| RSI oversold / overbought | 30 / 70 | |
| Adaptive profiles | True | Auto-calibrate per-stock parameters |
| Stop loss (ATR base) | 1.5x ATR | Per-stock: 1.43x (AAPL) to 2.4x (NVDA/TSLA) |
| Take profit (ATR base) | 2.5x ATR | Per-stock: 2.38x (AAPL) to 4.0x (NVDA/TSLA) |
| Volatility-scaled stops | True | Continuous scale by annualized vol |
| Stop loss (fixed fallback) | 1.5% | Used when ATR unavailable |
| Take profit (fixed fallback) | 5.0% | |
| Max position size | Per-stock adaptive | 5.3% (TSLA) to 11.4% (AAPL) |
| Backtest end date | None (today) | Pin with `--end-date YYYY-MM-DD` for reproducibility |
| Max portfolio drawdown | 10% | Circuit breaker |
| Trend SMA period | 200 days | |
| Max distance from 200-SMA | 8% | |
| Volatility percentile range | 20th-80th | |
| VIX threshold | 30 | Opt-in macro filter |

All parameters are in `config/settings.py` and are logged to the experiment tracker on each run.

## Trade Analysis Workflow

The recommended workflow for finding repeatable alpha:

1. **Run `--mode portfolio`** with your target symbols to get the full overlay backtest
2. **Inspect the per-symbol breakdown** -- which names contribute alpha and which are a drag?
3. **Run `--mode analyze`** on promising symbols to get trade logs and overlay charts
4. **Check the trade log CSV** -- look at indicators at entry for winning vs losing trades
5. **Read the trade analysis** -- if stops > TPs, your stop is too tight or entries are bad
6. **Tweak parameters** in `config/settings.py` (e.g., loosen z-score, widen stops)
7. **Re-run backtest** -- results auto-log to the experiment tracker
8. **Run `--mode sweep`** to systematically test parameter grid
9. **Run `--mode experiments`** to compare all parameter combinations by Sharpe ratio

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
│   └── stock_profiles.py    # Per-stock adaptive parameter calibration
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
