import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

SYMBOLS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
TRAINING_SYMBOLS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
LOOKBACK_PERIOD = 252
Z_SCORE_ENTRY_THRESHOLD = 2.0
Z_SCORE_EXIT_THRESHOLD = 0.3
RSI_OVERSOLD = 25
RSI_OVERBOUGHT = 70
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0
MAX_POSITION_SIZE_PCT = 0.03
STOP_LOSS_PCT = 0.012
TAKE_PROFIT_PCT = 0.06
MAX_PORTFOLIO_DRAWDOWN_PCT = 0.1
ML_CONFIDENCE_THRESHOLD = 0.45
ML_LOOKBACK_YEARS = 5
FORWARD_RETURN_PERIOD = 5
TRADING_INTERVAL_SECONDS = 60

# Strategy mode
USE_ML_FILTER = False  # Can be toggled off for pure statistical mode
USE_REGIME_DETECTION = False  # Enable regime-based position sizing

# ATR-based stops
ATR_STOP_MULTIPLIER = 1.25
ATR_PROFIT_MULTIPLIER = 2.0

# Volatility regime filter
VOL_PERCENTILE_LOW = 20   # Don't trade below this volatility percentile
VOL_PERCENTILE_HIGH = 80  # Don't trade above this volatility percentile

# Trend filter
USE_TREND_FILTER = True
TREND_SMA_PERIOD = 200

# Signal strength minimum
MIN_SIGNAL_STRENGTH = 0.35

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
