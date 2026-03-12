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
│  │  VIX /   │───▶│  4-Layer   │    │  ┌─────────────┐  ┌───────────────┐   │ │
│  │  Macro   │    │  Filter    │    │  │  Momentum   │  │   Adaptive    │   │ │
│  └──────────┘    │  Chain     │    │  │ (trend/ADX) │  │(regime switch)│   │ │
│                  └────────────┘    │  └─────────────┘  └───────────────┘   │ │
│                                    └───────────────────────────────────────┘ │
│                                                    │                         │
│  ┌──────────────────────┐    ┌────────────┐    ┌───▼───────────────────────┐ │
│  │  Regime Detector     │    │    Risk    │◀───│  Backtest / Execution     │ │
│  │  (GMM, 3 regimes)    │    │  Manager   │    │  + Trade Analysis         │ │
│  └──────────────────────┘    └────────────┘    │  + Experiment Tracker     │ │
│                                                └───────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Signal Generation & Filtering
- **Mean Reversion Signals**: Z-score, RSI, Bollinger Bands, and volume confirmation with weighted scoring
- **4-Layer Filter Chain**: Each filter logs how many signals it suppressed
  1. **Trend Filter** (200-day SMA) — buy dips only in uptrends, sell rips only in downtrends
  2. **Volatility Regime Filter** — trade only when 20-day vol is between 20th-80th percentile
  3. **Distance-from-Fair-Value Filter** — block entries when price is >8% from 200-SMA
  4. **Minimum Signal Strength** — require weighted confirmation score >= 0.28
- **VIX Macro Filter** (opt-in) — block all entries when VIX > 30

### Strategies
- **Mean Reversion** with multi-indicator confirmation and configurable looseness
- **Pairs Trading**: Market-neutral cointegration-based spread trading with risk controls
- **Momentum/Trend Following**: Multi-factor scoring with trailing ATR stops and rebalancing
- **Adaptive Regime Switching**: GMM-driven allocation between strategies
- **Combined Portfolio**: 50% momentum + 30% pairs + 20% cash reserve

### Trade Analysis & Alpha Discovery
- **Trade Log Export**: CSV with all indicators at entry (z-score, RSI, BB %B, volume, ATR, volatility, distance from SMA, MACD histogram, signal strength)
- **Trade Overlay Charts**: 4-panel visualization — price with buy/sell markers + Bollinger Bands + 200-SMA, z-score, RSI, cumulative P&L
- **Per-Trade Analysis**: Stop vs take-profit hit rates, expectancy per trade, P&L by exit reason, winner vs loser indicator comparison
- **Always-On Benchmark**: Every run compares return, Sharpe, and max drawdown against SPY buy-and-hold
- **Experiment Tracker**: CSV-based log of every parameter combination tested, sortable by Sharpe

### ML & Risk
- **ML Signal Filter**: LightGBM classifier with walk-forward validation (togglable with `--no-ml`)
- **ATR-Based Dynamic Stops**: Stop-loss = entry +/- 1.5x ATR; take-profit = entry +/- 2.5x ATR
- **Risk Management**: Fixed fractional sizing, max drawdown circuit breaker, regime-based position scaling
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
# Edit .env with your Alpaca paper trading API keys (optional — backtesting works without them)
```

## Usage

### Analyze trade quality (start here)

The `analyze` mode is the best way to understand *why* your system is or isn't generating alpha. It exports trade logs, generates overlay charts, and prints detailed winner-vs-loser indicator comparisons.

```bash
python main.py --mode analyze --symbols SPY NVDA
```

Output includes:
- `trade_log_SPY.csv` / `trade_log_NVDA.csv` — every trade with all indicators at entry
- `trades_overlay_SPY.png` — price chart with buy/sell markers, z-score, RSI, cumulative P&L
- Per-trade analysis: stop/TP hit rates, expectancy, indicator differences between winners and losers
- Actionable warnings (e.g., "stops hit more often than TPs — consider widening stop")

### Backtest with full reporting

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

Every backtest now prints:
- Performance report with benchmark comparison
- Trade analysis (stop/TP rates, expectancy, holding periods)
- P&L breakdown by exit reason
- Winner vs loser indicator comparison
- Results are automatically logged to the experiment tracker

### Compare all strategies

```bash
python main.py --mode compare --symbols SPY AAPL MSFT GOOGL NVDA
```

### View experiment history

```bash
python main.py --mode experiments
```

Output:
```
=== Experiment Tracker Summary ===
  experiment_id       strategy symbols  total_return  sharpe_ratio  max_drawdown  num_trades  win_rate   alpha  z_score_entry  min_signal_strength
20260312_010457 mean_reversion    NVDA        0.0882        0.3020       -0.3026           8     0.375 -0.9369            1.7                 0.28
20260312_010456 mean_reversion     SPY        0.0021        0.0605       -0.0678          12     0.500 -0.3390            1.7                 0.28
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

Buy when z-score < -1.7 and at least 1 of: RSI < 30, price near lower Bollinger Band, volume spike. Signal strength is a weighted combination of all confirmations.

| Filter | Logic |
|--------|-------|
| **Trend filter** | BUY only above 200-day SMA; SELL only below |
| **Volatility filter** | Only trade when 20-day vol is in 20th-80th percentile |
| **Distance filter** | Only enter when price is within 8% of 200-day SMA |
| **ATR stops** | Stop = entry +/- 1.5x ATR; TP = entry +/- 2.5x ATR |

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
| Z-score entry threshold | 1.7 | Lower = more trades, higher = fewer/stronger |
| Min signal strength | 0.28 | Weighted confirmation score minimum |
| Min optional confirmations | 1 | Beyond z-score (1=looser, 2=tighter) |
| RSI oversold / overbought | 30 / 70 | |
| Stop loss (ATR) | 1.5x ATR | |
| Take profit (ATR) | 2.5x ATR | |
| Stop loss (fixed) | 1.5% | Fallback when ATR unavailable |
| Take profit (fixed) | 5.0% | |
| Max position size | 3% of portfolio | |
| Max portfolio drawdown | 10% | Circuit breaker |
| Trend SMA period | 200 days | |
| Max distance from 200-SMA | 8% | |
| Volatility percentile range | 20th-80th | |
| VIX threshold | 30 | Opt-in macro filter |

All parameters are in `config/settings.py` and are logged to the experiment tracker on each run.

## Trade Analysis Workflow

The recommended workflow for finding repeatable alpha:

1. **Run `--mode analyze`** on your target symbols to get trade logs and overlay charts
2. **Inspect the trade log CSV** — look at indicators at entry for winning vs losing trades
3. **Check the overlay chart** — are you buying real dips or fighting the trend?
4. **Read the trade analysis** — if stops > TPs, your stop is too tight or entries are bad
5. **Tweak parameters** in `config/settings.py` (e.g., loosen z-score, widen stops)
6. **Re-run backtest** — results auto-log to the experiment tracker
7. **Run `--mode experiments`** to compare all parameter combinations by Sharpe ratio
8. **Repeat** until Sharpe > 0.5 and profit factor > 1.3 while keeping drawdown < 10%

## Project Structure

```
equities-mean-reversion-ml/
├── main.py                  # CLI orchestrator (backtest, analyze, compare, train, trade, experiments)
├── config/
│   └── settings.py          # All tunable parameters
├── data/
│   └── fetcher.py           # yfinance + Alpaca data fetching
├── features/
│   └── indicators.py        # Technical indicators (z-score, RSI, BB, MACD, ATR, VIX, etc.)
├── strategy/
│   ├── signals.py           # Mean reversion signal generator with 4-layer filter chain
│   ├── ml_filter.py         # LightGBM signal filter
│   ├── regime_detector.py   # Gaussian Mixture Model regime detection
│   ├── pairs_trading.py     # Cointegration-based pairs trading
│   ├── momentum.py          # Trend following with momentum scoring
│   └── adaptive.py          # Regime-switching strategy orchestrator
├── backtest/
│   └── engine.py            # Event-driven backtester with trade logging and analysis
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

145 tests run entirely on synthetic data — no API keys or network access required.

## Disclaimer

This software is for **educational and paper trading purposes only**. Past performance does not guarantee future results. Do not use with real money without thorough testing and understanding of the risks involved. The authors are not responsible for any financial losses.
