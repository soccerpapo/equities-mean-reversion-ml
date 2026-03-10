# Equities Mean Reversion ML Trading System

A production-ready algorithmic trading system that combines classical mean reversion strategies with optional machine learning enhancements. Supports three strategy modes — pure statistical, ML-filtered, and regime-aware — with a built-in comparison tool.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SYSTEM ARCHITECTURE                             │
│                                                                     │
│  ┌──────────┐    ┌────────────┐    ┌────────────────────────────┐  │
│  │  Data    │───▶│  Feature   │───▶│        Strategy            │  │
│  │ Fetcher  │    │  Engine    │    │  SignalGen (z-score, RSI,  │  │
│  │(yfinance)│    │(Indicators)│    │  Bollinger, Volume)        │  │
│  └──────────┘    └────────────┘    └──────────┬─────────────────┘  │
│                                               │                     │
│              ┌────────────────────────────────┤                     │
│              │                                │                     │
│  ┌───────────▼────────┐    ┌──────────────────▼────────────────┐   │
│  │  Regime Detector   │    │  ML Filter (optional)             │   │
│  │  (GMM, 3 regimes)  │    │  LightGBM signal classifier       │   │
│  │  Path B            │    │  Original                         │   │
│  └───────────┬────────┘    └──────────────────┬────────────────┘   │
│              └────────────────────────────────┘                     │
│                                               │                     │
│  ┌──────────┐    ┌────────────┐    ┌──────────▼─────────────────┐  │
│  │ Backtest │◀───│    Risk    │◀───│   Execution                │  │
│  │  Engine  │    │  Manager   │    │  AlpacaTrader              │  │
│  └──────────┘    └────────────┘    └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- **Mean Reversion Signals**: Z-score, RSI, Bollinger Bands, and volume confirmation with weighted signal strength
- **Trend Filter**: 200-day SMA filter — only buy dips in uptrends, sell rips in downtrends
- **Volatility Regime Filter**: Trade only when 20-day volatility percentile is between 20th–80th percentile
- **Path A — Pure Statistical**: No ML, uses ATR-based dynamic stops and the filters above
- **Path B — Regime Detection**: Gaussian Mixture Model classifies market into 3 regimes; position sizes scaled automatically
- **ML Signal Filter**: LightGBM classifier (original approach, togglable with `--no-ml`)
- **ATR-Based Dynamic Stops**: Stop-loss = entry ± 2×ATR; take-profit = entry ± 3×ATR
- **Risk Management**: Fixed fractional position sizing, stop-loss/take-profit, max drawdown circuit breaker
- **Paper Trading**: Full Alpaca API integration with bracket orders
- **Backtesting**: Event-driven engine with slippage modeling and performance reporting
- **Strategy Comparison**: `--mode compare` runs all three approaches side-by-side
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

### Backtest — Pure Statistical (Path A, no ML)

```bash
python main.py --mode backtest --no-ml
```

### Backtest — With Regime Detection (Path B)

```bash
python main.py --mode backtest --regime
```

### Backtest — With Regime Detection and no ML

```bash
python main.py --mode backtest --no-ml --regime
```

### Backtest — Original (ML filter enabled, default)

```bash
python main.py --mode backtest --symbol SPY
```

### Compare all three approaches on the same data

```bash
python main.py --mode compare --symbol SPY
```

Output looks like:
```
=== Strategy Comparison ===
Approach                 | Return     | Sharpe     | MaxDD      | Trades   | WinRate
---...
SPY: Pure Statistical    | X.XX%      | X.XX       | X.XX%      | XX       | XX.X%
SPY: + Regime Detection  | X.XX%      | X.XX       | X.XX%      | XX       | XX.X%
SPY: + ML Filter         | X.XX%      | X.XX       | X.XX%      | XX       | XX.X%
SPY: Buy & Hold SPY      | X.XX%      | —          | —          | 1        | —
```

### Train ML model

```bash
python main.py --mode train
```

### Live paper trading — pure statistical (best approach)

```bash
python main.py --mode trade --no-ml --regime
```

### Live paper trading — default (with ML filter)

```bash
python main.py --mode trade
```

## Strategy

### Signal Generation

Buy signals are generated when:
- Z-score of close price crosses below **-1.5** (oversold)
- At least 1 of: RSI < **30**, price near lower Bollinger Band (`%B < 0.1`), or volume z-score > 1.0

Sell signals are the mirror image. All conditions use only historical data — no look-ahead bias.

Signal strength is a weighted combination:
- Z-score confirmation: **30%**
- RSI: **25%**
- Bollinger Band: **25%**
- Volume: **20%**

Only signals with `signal_strength ≥ 0.5` are taken.

### Path A: Pure Statistical Filters

| Filter | Logic |
|--------|-------|
| **Trend filter** | BUY only above 200-day SMA; SELL only below 200-day SMA |
| **Volatility filter** | Only trade when 20-day vol is between 20th–80th percentile of last 252 days |
| **ATR stops** | Stop-loss = entry ± 2×ATR; take-profit = entry ± 3×ATR |

### Path B: Regime Detection

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

### ML Filter (original)

A LightGBM classifier is trained on a 80/20 time-series split to predict whether the next 5-day forward return is positive. Only signals with confidence ≥ **0.55** pass through to execution.

### Risk Controls

| Parameter | Default |
|-----------|---------|
| Max position size | 10% of portfolio |
| Stop loss (fixed) | 2% per trade |
| Take profit (fixed) | 4% per trade |
| Stop loss (ATR) | entry ± 2×ATR |
| Take profit (ATR) | entry ± 3×ATR |
| Max portfolio drawdown | 10% |
| Max open positions | 5 |
| Min signal strength | 0.5 |

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests run entirely on synthetic data — no API keys or network access required.

## Disclaimer

This software is for **educational and paper trading purposes only**. Past performance does not guarantee future results. Do not use with real money without thorough testing and understanding of the risks involved. The authors are not responsible for any financial losses.
