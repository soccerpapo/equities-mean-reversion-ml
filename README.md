# Equities Mean Reversion ML Trading System

A production-ready algorithmic trading system that combines classical mean reversion strategies with machine learning enhancements and three new adaptive strategies: **Pairs Trading**, **Momentum/Trend Following**, and **Adaptive Regime Switching**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                                  │
│                                                                               │
│  ┌──────────┐    ┌────────────┐    ┌──────────────────────────────────────┐  │
│  │  Data    │───▶│  Feature   │───▶│            Strategy Layer            │  │
│  │ Fetcher  │    │  Engine    │    │                                      │  │
│  │(yfinance)│    │(Indicators)│    │  ┌─────────────┐  ┌───────────────┐  │  │
│  └──────────┘    └────────────┘    │  │ Mean Revert │  │ Pairs Trading │  │  │
│                                    │  │  + ML Filter│  │(cointegration)│  │  │
│                                    │  └─────────────┘  └───────────────┘  │  │
│                                    │  ┌─────────────┐  ┌───────────────┐  │  │
│                                    │  │  Momentum   │  │   Adaptive    │  │  │
│                                    │  │ (trend/ADX) │  │(regime switch)│  │  │
│                                    │  └─────────────┘  └───────────────┘  │  │
│                                    └──────────────────────────────────────┘  │
│                                                    │                          │
│  ┌──────────────────────┐    ┌────────────┐    ┌───▼──────────────────────┐  │
│  │  Regime Detector     │    │    Risk    │◀───│   Backtest / Execution   │  │
│  │  (GMM, 3 regimes)    │    │  Manager   │    │   AlpacaTrader           │  │
│  └──────────────────────┘    └────────────┘    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Mean Reversion Signals**: Z-score, RSI, Bollinger Bands, and volume confirmation
- **Trend Filter**: 200-day SMA filter — only buy dips in uptrends, sell rips in downtrends
- **Volatility Regime Filter**: Trade only when volatility is between 20th–80th percentile
- **ML Signal Filter**: LightGBM classifier trained on multiple symbols (togglable with `--no-ml`)
- **Pairs Trading** ⭐: Market-neutral cointegration-based mean reversion
- **Momentum/Trend Following** ⭐: Multi-factor momentum scoring with trailing ATR stops
- **Adaptive Strategy** ⭐: Regime detection drives dynamic allocation between all strategies
- **ATR-Based Dynamic Stops**: Stop-loss = entry ± 2×ATR; take-profit = entry ± 3×ATR
- **Risk Management**: Fixed fractional sizing, stop-loss, take-profit, max drawdown circuit breaker
- **Paper Trading**: Full Alpaca API integration with bracket and trailing stop orders
- **Backtesting**: Event-driven engine with slippage modeling across all six strategies
- **Strategy Comparison**: `--mode compare` runs all six approaches side-by-side
- **No look-ahead bias**: All indicators computed using only past data

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd equities-mean-reversion-ml
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Alpaca paper trading API keys
```

### 3. Get Alpaca API Keys

Sign up at [https://alpaca.markets](https://alpaca.markets) and create paper trading API keys.

## Usage

### Compare all strategies head-to-head

```bash
python main.py --mode compare
```

Output:
```
=== Strategy Comparison ===
Approach                 | Return     | Sharpe     | MaxDD      | Trades   | WinRate
-------------------------+------------+------------+------------+----------+-----------
Pure Mean Reversion      | X.XX%      | X.XX       | X.XX%      | XX       | XX.X%
+ Regime Detection       | X.XX%      | X.XX       | X.XX%      | XX       | XX.X%
+ ML Filter              | X.XX%      | X.XX       | X.XX%      | XX       | XX.X%
Pairs Trading            | X.XX%      | X.XX       | X.XX%      | XX       | XX.X%
Momentum                 | X.XX%      | X.XX       | X.XX%      | XX       | XX.X%
Adaptive (Regime Switch) | X.XX%      | X.XX       | X.XX%      | XX       | XX.X%
Buy & Hold SPY           | X.XX%      | —          | —          | 1        | —
```

### Backtest specific strategy

```bash
# Backtest all strategies (default)
python main.py --mode backtest --strategy all

# Backtest pairs trading only
python main.py --mode backtest --strategy pairs

# Backtest momentum only
python main.py --mode backtest --strategy momentum

# Backtest adaptive (regime-switching) only
python main.py --mode backtest --strategy adaptive

# Backtest mean reversion only (original)
python main.py --mode backtest --strategy mean_reversion
```

### Paper trade with adaptive strategy (recommended)

```bash
python main.py --mode trade --strategy adaptive
```

### Train ML model

```bash
python main.py --mode train
```

### Other useful commands

```bash
# Pure statistical mean reversion (no ML)
python main.py --mode backtest --no-ml --strategy mean_reversion

# With regime detection
python main.py --mode backtest --regime --strategy mean_reversion

# Multi-symbol backtest
python main.py --mode backtest --symbols SPY AAPL MSFT GOOGL
```

## Strategies

### 1. Mean Reversion (Original)

**Signal Generation**: Buy when z-score < -1.5 and at least one of: RSI < 30, price near lower Bollinger Band, volume spike. Signal strength is a weighted combination of all confirmations.

| Filter | Logic |
|--------|-------|
| **Trend filter** | BUY only above 200-day SMA; SELL only below 200-day SMA |
| **Volatility filter** | Only trade when 20-day vol is between 20th–80th percentile |
| **ATR stops** | Stop-loss = entry ± 2×ATR; take-profit = entry ± 3×ATR |

### 2. Pairs Trading (Market-Neutral) ⭐

Identifies **cointegrated** pairs of stocks using the Engle-Granger test and trades the spread between them. Because both legs offset each other, this strategy is **market-neutral** — profits come from mean reversion of the spread regardless of overall market direction.

| Condition | Action |
|-----------|--------|
| Spread z-score < -2.0 | BUY spread (buy stock A, sell stock B) |
| Spread z-score > +2.0 | SELL spread (sell stock A, buy stock B) |
| \|Z-score\| < 0.5 | CLOSE position (spread reverted) |
| \|Z-score\| > 3.0 | STOP LOSS (spread diverging) |

- **Hedge ratio**: Determined by OLS regression to ensure dollar-neutral positions
- **Rolling hedge**: Recalculated on a 60-day window to adapt to changing relationships
- **Cointegration tested on expanding window**: Pairs are re-evaluated to detect breakdown

### 3. Momentum / Trend Following ⭐

Ranks stocks by **composite momentum score** (weighted blend of 1M, 3M, 6M, and 12M returns) and enters long positions in the top-N stocks when trend indicators confirm.

**Entry conditions** (all must be true):
- Momentum score > 0.3
- Price > 200-day SMA
- ADX > 25 (strong trend)

**Exit conditions** (any triggers exit):
- Momentum score < -0.3
- Price < 200-day SMA
- Trailing stop hit (2×ATR trailing stop)

The **trailing stop** locks in profits as the stock rises and exits automatically when momentum fades.

### 4. Adaptive Strategy (Regime-Based Switching) ⭐

Uses the **Gaussian Mixture Model regime detector** to identify the current market environment and dynamically allocates capital to the best-suited strategy.

| Regime | Description | Allocation |
|--------|-------------|------------|
| **0** | Low volatility / mean-reverting | 70% pairs, 20% momentum, 10% cash |
| **1** | Normal / trending | 20% pairs, 70% momentum, 10% cash |
| **2** | High volatility / crisis | 100% cash (no trading) |

**Smooth transitions**: When regime changes, positions are reduced gradually over 3 days to avoid whipsaw. This is the **recommended default strategy** for live trading.

### 5. Regime Detection (Path B)

A **Gaussian Mixture Model** classifies each day into one of 3 regimes using:
- 20-day realized volatility (annualised)
- Volatility of volatility
- Average absolute daily return
- Volume ratio (5-day vs 20-day)
- Return autocorrelation (negative = mean-reverting)
- 20-day kurtosis

| Regime | Description | Position Multiplier |
|--------|-------------|---------------------|
| **0** | Low volatility / mean-reverting | **1.0** (full size) |
| **1** | Normal / trending | **0.5** (half size) |
| **2** | High volatility / crisis | **0.0** (no trading) |

### 6. ML Filter (Path C)

A LightGBM classifier trained across 8 symbols on 5 years of data. Predicts whether the next 3-day forward return is positive. Only signals with confidence ≥ **0.55** pass through to execution.

## Risk Controls

| Parameter | Default |
|-----------|---------|
| Max position size | 10% of portfolio |
| Stop loss (fixed) | 2% per trade |
| Take profit (fixed) | 4% per trade |
| Stop loss (ATR) | entry ± 2×ATR |
| Take profit (ATR) | entry ± 3×ATR |
| Max portfolio drawdown | 10% |
| Pairs z-score stop | ±3.0 |
| Pairs lookback window | 60 days |
| Momentum trailing stop | 2×ATR |
| Regime transition period | 3 days |

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests run entirely on synthetic data — no API keys or network access required. New tests cover:
- `tests/test_pairs.py` — cointegration detection, spread z-score, signal generation, hedge ratio
- `tests/test_momentum.py` — momentum scoring, trend detection, ranking, trailing stops
- `tests/test_adaptive.py` — regime-to-strategy mapping, capital allocation, transition logic

## Disclaimer

This software is for **educational and paper trading purposes only**. Past performance does not guarantee future results. Do not use with real money without thorough testing and understanding of the risks involved. The authors are not responsible for any financial losses.
