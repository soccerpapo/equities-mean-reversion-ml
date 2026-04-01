import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

SYMBOLS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
TRAINING_SYMBOLS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
LOOKBACK_PERIOD = 252
Z_SCORE_ENTRY_THRESHOLD = 2.0
Z_SCORE_EXIT_THRESHOLD = 0.3
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0
MAX_POSITION_SIZE_PCT = 0.10
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.06
MAX_PORTFOLIO_DRAWDOWN_PCT = 0.1
ML_CONFIDENCE_THRESHOLD = 0.45
ML_LOOKBACK_YEARS = 5
FORWARD_RETURN_PERIOD = 5
TRADING_INTERVAL_SECONDS = 60

# Long-only mode: suppress all SELL / short signals
LONG_ONLY = True

# Benchmark overlay: invest idle cash in the benchmark (SPY) when not in a
# mean-reversion trade.  This earns the market return on uninvested capital
# so the strategy only needs to beat the market on the trades it takes.
USE_BENCHMARK_OVERLAY = True
BENCHMARK_OVERLAY_SYMBOL = "SPY"

# Strategy mode
USE_ML_FILTER = False
USE_REGIME_DETECTION = False

# ATR-based stops (base multipliers — scaled by per-asset volatility when
# USE_VOLATILITY_SCALED_STOPS is True)
ATR_STOP_MULTIPLIER = 2.0
ATR_PROFIT_MULTIPLIER = 3.0
USE_VOLATILITY_SCALED_STOPS = True

# Volatility regime filter
USE_VOLATILITY_FILTER = True
VOL_PERCENTILE_LOW = 20
VOL_PERCENTILE_HIGH = 80

# Trend filter
USE_TREND_FILTER = True
TREND_SMA_PERIOD = 200

# Signal strength minimum
MIN_SIGNAL_STRENGTH = 0.0

# Distance from fair value filter (max % distance from 200-SMA to allow entry)
USE_DIST_SMA200_FILTER = True
MAX_DIST_SMA200 = 0.10

# VIX macro filter
USE_VIX_FILTER = False
VIX_THRESHOLD_HIGH = 30

# Min confirmations required beyond z-score (1 = looser, 2 = tighter)
MIN_OPTIONAL_CONFIRMATIONS = 1

# Backtest reproducibility: pin the end date so the data window doesn't shift
# when re-running on a different calendar day.  Empty string = use today (default).
# Set via CLI: --end-date 2026-03-19
BACKTEST_END_DATE = ""

# Per-stock adaptive profiles: auto-calibrate parameters from historical data
USE_STOCK_PROFILES = True

# Manual per-stock overrides (applied on top of auto-calibration).
# Keys are symbols; values are dicts of parameter overrides.
# Example: {"NVDA": {"atr_stop_mult": 2.2, "max_position_size_pct": 0.08}}
STOCK_PROFILE_OVERRIDES: dict = {}

# Regime detection
REGIME_N_COMPONENTS = 3  # Number of regimes to detect

# Pairs Trading
PAIRS_ZSCORE_ENTRY = 2.0
PAIRS_ZSCORE_EXIT = 0.5
PAIRS_ZSCORE_STOP = 3.5
PAIRS_LOOKBACK = 60  # Rolling window for spread z-score
PAIRS_COINT_PVALUE = 0.05  # Max p-value for cointegration
CAPITAL_PER_PAIR = 0.03  # 3% of portfolio per pair
MAX_SIMULTANEOUS_PAIRS = 3  # Maximum number of open pair positions at once
MAX_PAIR_LOSS_PCT = 0.03  # Close a pair if unrealized loss exceeds 3% of capital_per_pair
PAIR_COOLDOWN_DAYS = 5  # Trading days before a stopped-out pair can re-enter
MAX_PORTFOLIO_EXPOSURE = 0.15  # Max fraction of portfolio committed to pairs at any time

# Momentum
MOMENTUM_SMA_FAST = 50
MOMENTUM_SMA_SLOW = 200
MOMENTUM_ADX_THRESHOLD = 25
MOMENTUM_TOP_N = 4  # Number of top stocks to hold
TRAILING_STOP_ATR_MULT = 1.5
MOMENTUM_REBALANCE_DAYS = 20  # Rebalance portfolio every N trading days

# Adaptive Strategy
REGIME_ALLOCATIONS = {
    0: {"pairs": 0.7, "momentum": 0.2, "cash": 0.1},  # Mean-reverting
    1: {"pairs": 0.2, "momentum": 0.7, "cash": 0.1},  # Trending
    2: {"pairs": 0.0, "momentum": 0.0, "cash": 1.0},  # Crisis
}
TRANSITION_DAYS = 5
