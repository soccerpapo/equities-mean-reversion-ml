import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

SYMBOLS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
TRAINING_SYMBOLS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
LOOKBACK_PERIOD = 252
Z_SCORE_ENTRY_THRESHOLD = 1.5
Z_SCORE_EXIT_THRESHOLD = 0.5
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0
MAX_POSITION_SIZE_PCT = 0.1
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
MAX_PORTFOLIO_DRAWDOWN_PCT = 0.1
ML_CONFIDENCE_THRESHOLD = 0.55
ML_LOOKBACK_YEARS = 5
FORWARD_RETURN_PERIOD = 3
TRADING_INTERVAL_SECONDS = 60

# Strategy mode
USE_ML_FILTER = True  # Can be toggled off for pure statistical mode
USE_REGIME_DETECTION = True  # Enable regime-based position sizing

# ATR-based stops
ATR_STOP_MULTIPLIER = 2.0
ATR_PROFIT_MULTIPLIER = 3.0

# Volatility regime filter
VOL_PERCENTILE_LOW = 20   # Don't trade below this volatility percentile
VOL_PERCENTILE_HIGH = 80  # Don't trade above this volatility percentile

# Trend filter
USE_TREND_FILTER = True
TREND_SMA_PERIOD = 200

# Signal strength minimum
MIN_SIGNAL_STRENGTH = 0.5

# Regime detection
REGIME_N_COMPONENTS = 3  # Number of regimes to detect
