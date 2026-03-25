import logging
import time
from typing import Dict, Optional
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _period_to_start_date(period: str, end_date: str) -> str:
    """Convert a relative period string + absolute end date to an absolute start date.

    Supports yfinance-style period strings: '1y', '2y', '5y', '6mo', '30d', etc.
    """
    end_dt = pd.Timestamp(end_date)
    if period.endswith("y"):
        start_dt = end_dt - pd.DateOffset(years=int(period[:-1]))
    elif period.endswith("mo"):
        start_dt = end_dt - pd.DateOffset(months=int(period[:-2]))
    elif period.endswith("d"):
        start_dt = end_dt - pd.DateOffset(days=int(period[:-1]))
    else:
        raise ValueError(f"Cannot parse period '{period}' with pinned end_date")
    return start_dt.strftime("%Y-%m-%d")


class DataFetcher:
    """Fetches market data from yfinance and Alpaca."""

    def __init__(self):
        self._alpaca = None
        self._init_alpaca()

    def _init_alpaca(self):
        """Initialize Alpaca client if API keys are available."""
        try:
            from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
            if ALPACA_API_KEY and ALPACA_SECRET_KEY:
                import alpaca_trade_api as tradeapi
                self._alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
                logger.info("Alpaca client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Alpaca client: {e}")

    def fetch_historical(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch historical OHLCV data using yfinance.

        Args:
            symbol: Ticker symbol
            period: Data period (e.g., '1y', '6mo')
            interval: Data interval (e.g., '1d', '1h')

        Returns:
            DataFrame with standardized column names
        """
        from config import settings
        pinned_end = getattr(settings, "BACKTEST_END_DATE", "")

        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol)
                if pinned_end:
                    start_date = _period_to_start_date(period, pinned_end)
                    # end is exclusive in yfinance, add 1 day so the pinned date is included
                    end_dt = pd.Timestamp(pinned_end) + pd.DateOffset(days=1)
                    df = ticker.history(start=start_date, end=end_dt.strftime("%Y-%m-%d"), interval=interval)
                else:
                    df = ticker.history(period=period, interval=interval)
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                df.index = df.index.tz_localize(None)
                df.dropna(inplace=True)
                logger.info(f"Fetched {len(df)} rows for {symbol}")
                return df
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {symbol}: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        logger.error(f"All attempts failed for {symbol}")
        return pd.DataFrame()

    def fetch_multiple(self, symbols: list, period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            period: Data period
            interval: Data interval

        Returns:
            Dict mapping symbol to DataFrame
        """
        results = {}
        for symbol in symbols:
            df = self.fetch_historical(symbol, period=period, interval=interval)
            if not df.empty:
                results[symbol] = df
            time.sleep(0.1)
        return results

    def fetch_realtime(self, symbol: str) -> Optional[Dict]:
        """Get latest quote via Alpaca API.

        Args:
            symbol: Ticker symbol

        Returns:
            Dict with latest quote data, or None if unavailable
        """
        if self._alpaca is None:
            logger.warning("Alpaca client not initialized; cannot fetch realtime data")
            return None
        try:
            quote = self._alpaca.get_latest_quote(symbol)
            return {
                "symbol": symbol,
                "ask_price": quote.ap,
                "bid_price": quote.bp,
                "ask_size": quote.as_,
                "bid_size": quote.bs,
            }
        except Exception as e:
            logger.error(f"Error fetching realtime quote for {symbol}: {e}")
            return None

    def fetch_account_info(self) -> Optional[Dict]:
        """Get account balance from Alpaca.

        Returns:
            Dict with account info, or None if unavailable
        """
        if self._alpaca is None:
            logger.warning("Alpaca client not initialized; cannot fetch account info")
            return None
        try:
            account = self._alpaca.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return None
