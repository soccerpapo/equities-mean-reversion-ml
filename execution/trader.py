import logging
from typing import Dict, List, Optional
from config import settings

logger = logging.getLogger(__name__)


class AlpacaTrader:
    """Handles live paper trading via Alpaca API."""

    def __init__(self):
        """Initialize Alpaca REST client for paper trading."""
        self._api = None
        self._init_client()

    def _init_client(self):
        """Initialize Alpaca client if API keys are available."""
        try:
            if settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY:
                import alpaca_trade_api as tradeapi
                self._api = tradeapi.REST(
                    settings.ALPACA_API_KEY,
                    settings.ALPACA_SECRET_KEY,
                    settings.ALPACA_BASE_URL,
                )
                logger.info("Alpaca trader initialized")
            else:
                logger.warning("Alpaca API keys not set. Trader in dry-run mode.")
        except Exception as e:
            logger.warning(f"Could not initialize Alpaca client: {e}")

    def _require_api(self):
        """Raise RuntimeError if Alpaca API is not initialized."""
        if self._api is None:
            raise RuntimeError("Alpaca API client not initialized. Set API keys in .env.")

    def get_account(self) -> Optional[Dict]:
        """Return account info.

        Returns:
            Dict with equity, cash, buying_power, status or None on error
        """
        try:
            self._require_api()
            acct = self._api.get_account()
            return {
                "equity": float(acct.equity),
                "cash": float(acct.cash),
                "buying_power": float(acct.buying_power),
                "status": acct.status,
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None

    def get_positions(self) -> List[Dict]:
        """Return current open positions.

        Returns:
            List of position dicts
        """
        try:
            self._require_api()
            positions = self._api.list_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "side": p.side,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def place_order(
        self, symbol: str, qty: int, side: str, order_type: str = "market"
    ) -> Optional[Dict]:
        """Submit a market or limit order.

        Args:
            symbol: Ticker symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'

        Returns:
            Order dict or None on failure
        """
        try:
            self._require_api()
            order = self._api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force="day",
            )
            logger.info(f"Order placed: {side} {qty} {symbol} [{order_type}] id={order.id}")
            return {"id": order.id, "symbol": symbol, "qty": qty, "side": side}
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def place_bracket_order(
        self, symbol: str, qty: int, side: str, take_profit_price: float, stop_loss_price: float
    ) -> Optional[Dict]:
        """Submit a bracket order with take profit and stop loss.

        Args:
            symbol: Ticker symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            take_profit_price: Target exit price
            stop_loss_price: Stop loss price

        Returns:
            Order dict or None on failure
        """
        try:
            self._require_api()
            order = self._api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="gtc",
                order_class="bracket",
                take_profit={"limit_price": str(take_profit_price)},
                stop_loss={"stop_price": str(stop_loss_price)},
            )
            logger.info(f"Bracket order placed: {side} {qty} {symbol}")
            return {"id": order.id, "symbol": symbol, "qty": qty, "side": side}
        except Exception as e:
            logger.error(f"Error placing bracket order: {e}")
            return None

    def close_position(self, symbol: str) -> bool:
        """Close a specific position.

        Args:
            symbol: Ticker symbol

        Returns:
            True on success, False on failure
        """
        try:
            self._require_api()
            self._api.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Liquidate all open positions.

        Returns:
            True on success, False on failure
        """
        try:
            self._require_api()
            self._api.close_all_positions()
            logger.info("All positions closed")
            return True
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False

    def get_order_history(self) -> List[Dict]:
        """Retrieve past orders.

        Returns:
            List of order dicts
        """
        try:
            self._require_api()
            orders = self._api.list_orders(status="all", limit=100)
            return [
                {
                    "id": o.id,
                    "symbol": o.symbol,
                    "qty": float(o.qty),
                    "side": o.side,
                    "status": o.status,
                    "filled_avg_price": float(o.filled_avg_price or 0),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return []

    def is_market_open(self) -> bool:
        """Check if market is currently open.

        Returns:
            True if market is open
        """
        try:
            self._require_api()
            clock = self._api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False

    def place_pairs_order(
        self,
        symbol_a: str,
        qty_a: int,
        side_a: str,
        symbol_b: str,
        qty_b: int,
        side_b: str,
    ) -> Optional[Dict]:
        """Execute both legs of a pairs trade atomically.

        Submits orders for both legs sequentially.  If the second leg fails
        the first leg has already been submitted; callers should monitor for
        partial fills.

        Args:
            symbol_a: First leg ticker.
            qty_a: Quantity for the first leg.
            side_a: 'buy' or 'sell' for the first leg.
            symbol_b: Second leg ticker.
            qty_b: Quantity for the second leg.
            side_b: 'buy' or 'sell' for the second leg.

        Returns:
            Dict with 'leg_a' and 'leg_b' order dicts, or None on failure.
        """
        try:
            self._require_api()
            order_a = self.place_order(symbol_a, qty_a, side_a)
            order_b = self.place_order(symbol_b, qty_b, side_b)
            if order_a is None or order_b is None:
                logger.warning("Pairs order partially failed: leg_a=%s, leg_b=%s", order_a, order_b)
                return None
            logger.info(
                "Pairs order placed: %s %d %s / %s %d %s",
                side_a, qty_a, symbol_a, side_b, qty_b, symbol_b,
            )
            return {"leg_a": order_a, "leg_b": order_b}
        except Exception as e:
            logger.error(f"Error placing pairs order: {e}")
            return None

    def place_trailing_stop(
        self, symbol: str, qty: int, trail_pct: float
    ) -> Optional[Dict]:
        """Submit a trailing stop order to protect momentum profits.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares.
            trail_pct: Trailing percentage (e.g. 0.02 for 2%).

        Returns:
            Order dict or None on failure.
        """
        try:
            self._require_api()
            trail_price = round(trail_pct * 100, 2)  # Alpaca expects percentage as numeric
            order = self._api.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="trailing_stop",
                time_in_force="gtc",
                trail_percent=str(trail_price),
            )
            logger.info(f"Trailing stop placed: sell {qty} {symbol} trail={trail_pct:.2%}")
            return {"id": order.id, "symbol": symbol, "qty": qty, "side": "sell", "trail_pct": trail_pct}
        except Exception as e:
            logger.error(f"Error placing trailing stop: {e}")
            return None

    def get_portfolio_value(self) -> Optional[float]:
        """Return the current total portfolio value (equity).

        Returns:
            Portfolio equity as a float, or None on failure.
        """
        acct = self.get_account()
        if acct is not None:
            return acct.get("equity")
        return None
