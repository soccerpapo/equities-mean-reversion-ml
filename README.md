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

### Portfolio Backtest (2-year, 7 symbols on SPY overlay)

| Metric | Strategy | SPY Buy & Hold | Delta |
|--------|----------|----------------|-------|
| **Return** | **39.92%** | 31.79% | **+8.13%** |
| **Alpha** | **+8.13%** | -- | -- |
| **Sharpe** | **1.075** | 0.928 | **+0.147** |
| **Sortino** | **1.351** | -- | -- |
| Max Drawdown | -19.41% | -18.76% | -0.65% |
| Trades | 18 | -- | -- |
| Win Rate | 61.1% | -- | -- |
| Profit Factor | 3.40 | -- | -- |
| Expectancy | $658/trade | -- | -- |
| Capital Utilization | 44.3% | 100% | -- |

### Per-Symbol Alpha Breakdown

| Symbol | Trades | P&L | Win Rate |
|--------|--------|-----|----------|
| NVDA | 2 | +$4,517 | 100% |
| TSLA | 2 | +$3,784 | 100% |
| AAPL | 2 | +$2,767 | 100% |
| AMZN | 2 | +$755 | 50% |
| GOOGL | 2 | +$589 | 50% |
| META | 4 | +$528 | 50% |
| MSFT | 4 | -$1,089 | 25% |

### How we got here

The system went through three phases of improvement:

**Phase 1: Risk control** -- Long-only mode, 5-layer filter chain, volatility-scaled ATR stops, parameter sweep. Result: near-zero drawdown but negative alpha (sitting in cash 95% of the time while the market rallied).

**Phase 2: Capital utilization** -- Three structural changes solved the alpha problem:

1. **Benchmark overlay** -- Idle cash is invested in SPY instead of earning 0%. When a mean-reversion signal fires, SPY shares are liquidated to fund the trade. On exit, proceeds return to SPY. The strategy earns market return + incremental alpha from trades.
2. **Multi-symbol portfolio engine** -- A single portfolio holds SPY as its base and trades mean-reversion dips across 7 symbols concurrently. More symbols = more dip opportunities = higher capital utilization (15% -> 44%).
3. **Larger position sizing** -- Increased from 5% to 12% per trade. Drawdowns were well-controlled, leaving room to amplify winning trades.

**Phase 3 (current):** Alpha is positive and diversified across multiple names. The strategy beats SPY buy-and-hold by +8.13% with a Sharpe of 1.075.

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

### Trade Analysis & Alpha Discovery
- **Trade Log Export**: CSV with all indicators at entry (z-score, RSI, BB %B, volume, ATR, volatility, distance from SMA, MACD histogram, signal strength)
- **Trade Overlay Charts**: 4-panel visualization -- price with buy/sell markers + Bollinger Bands + 200-SMA, z-score, RSI, cumulative P&L
- **Per-Trade Analysis**: Stop vs take-profit hit rates, expectancy per trade, P&L by exit reason, winner vs loser indicator comparison
- **Always-On Benchmark**: Every run compares return, Sharpe, and max drawdown against SPY buy-and-hold
- **Experiment Tracker**: CSV-based log of every parameter combination tested, sortable by Sharpe
- **Parameter Sweep**: Automated grid search over z-score and signal strength thresholds with full logging

### Risk Management
- **ML Signal Filter**: LightGBM classifier with walk-forward validation (togglable with `--no-ml`)
- **Volatility-Scaled ATR Stops**: Base stop = 1.5x ATR, base TP = 2.5x ATR; automatically widened up to 1.6x for high-volatility names
- **Position Sizing**: 12% of portfolio per trade, scaled by signal strength and volatility
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
python main.py --mode portfolio --symbols SPY NVDA AAPL MSFT GOOGL AMZN META TSLA --years 2
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
| Stop loss (ATR base) | 1.5x ATR | Scaled up to 1.6x for high-vol names |
| Take profit (ATR base) | 2.5x ATR | Scaled up to 1.6x for high-vol names |
| Volatility-scaled stops | True | Auto-widen stops for high-vol assets |
| Stop loss (fixed fallback) | 1.5% | Used when ATR unavailable |
| Take profit (fixed fallback) | 5.0% | |
| Max position size | 12% of portfolio | Scaled by signal strength and volatility |
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
├── main.py                  # CLI (portfolio, backtest, analyze, compare, sweep, train, trade, experiments)
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
│   └── experiment_tracker.py # CSV-based parameter experiment logging
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
