# Equities Mean Reversion ML Trading System

A production-ready algorithmic trading system that combines classical mean reversion strategies with a LightGBM machine learning signal filter. Designed for paper trading via the Alpaca API with a robust backtesting engine.

```
┌─────────────────────────────────────────────────────────┐
│              SYSTEM ARCHITECTURE                        │
│                                                         │
│  ┌──────────┐    ┌────────────┐    ┌─────────────────┐  │
│  │  Data    │───▶│  Feature   │───▶│    Strategy     │  │
│  │ Fetcher  │    │  Engine    │    │  SignalGen +    │  │
│  │(yfinance)│    │(Indicators)│    │  ML Filter      │  │
│  └──────────┘    └────────────┘    └────────┬────────┘  │
│                                             │           │
│  ┌──────────┐    ┌────────────┐    ┌────────▼────────┐  │
│  │ Backtest │◀───│    Risk    │◀───│   Execution     │  │
│  │  Engine  │    │  Manager   │    │  AlpacaTrader   │  │
│  └──────────┘    └────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Features

- **Mean Reversion Signals**: Z-score, RSI, Bollinger Bands, and volume confirmation
- **ML Signal Filter**: LightGBM classifier trained on historical data to filter low-quality signals
- **Risk Management**: Fixed fractional position sizing, stop-loss/take-profit, max drawdown circuit breaker
- **Paper Trading**: Full Alpaca API integration with bracket orders
- **Backtesting**: Event-driven engine with slippage modeling and performance reporting
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

### Backtest mode (no API keys required)

```bash
python main.py --mode backtest --symbol SPY
```

### Train ML model

```bash
python main.py --mode train --symbol SPY
```

### Live paper trading

```bash
python main.py --mode trade
```

## Strategy

### Signal Generation

Buy signals are generated when:
- Z-score of close price crosses below **-2.0** (oversold)
- RSI < **30** (oversold confirmation)
- Price near or below lower Bollinger Band (`%B < 0.1`)
- Volume z-score > 1.0 (elevated volume confirmation)

Sell signals are the mirror image. All conditions use only historical data — no look-ahead bias.

### ML Filter

A LightGBM classifier is trained on a 80/20 time-series split to predict whether the next 5-day forward return is positive. Only signals with confidence ≥ **0.6** pass through to execution.

### Risk Controls

| Parameter | Default |
|-----------|---------|
| Max position size | 10% of portfolio |
| Stop loss | 2% per trade |
| Take profit | 4% per trade |
| Max portfolio drawdown | 10% |
| Max open positions | 5 |

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests run entirely on synthetic data — no API keys or network access required.

## Disclaimer

This software is for **educational and paper trading purposes only**. Past performance does not guarantee future results. Do not use with real money without thorough testing and understanding of the risks involved. The authors are not responsible for any financial losses.
