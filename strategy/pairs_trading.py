"""Pairs Trading strategy using cointegration-based mean reversion.

This module implements a market-neutral pairs trading strategy that:
  1. Finds cointegrated pairs using the Engle-Granger test
  2. Calculates the spread and its z-score
  3. Generates buy/sell signals when the spread diverges from the mean
  4. Uses stop-loss when the spread diverges beyond a threshold
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint

from config import settings

logger = logging.getLogger(__name__)


class PairsTrader:
    """Market-neutral pairs trading strategy based on cointegration.

    Identifies cointegrated pairs of stocks and trades their spread:
      - When spread z-score < -ENTRY: BUY spread (buy A, sell B)
      - When spread z-score > +ENTRY: SELL spread (sell A, buy B)
      - When |z-score| < EXIT: CLOSE position (mean reversion achieved)
      - When |z-score| > STOP: STOP LOSS (spread diverging, not reverting)
    """

    SIGNAL_BUY = 1      # Buy spread (buy A, sell B)
    SIGNAL_SELL = -1    # Sell spread (sell A, buy B)
    SIGNAL_CLOSE = 2    # Close existing position
    SIGNAL_STOP = -2    # Stop loss — close at a loss
    SIGNAL_NONE = 0     # No action

    def __init__(
        self,
        zscore_entry: float = None,
        zscore_exit: float = None,
        zscore_stop: float = None,
        lookback: int = None,
        coint_pvalue: float = None,
    ):
        """Initialise PairsTrader with configurable thresholds.

        Args:
            zscore_entry: Z-score threshold to enter a trade (default from settings).
            zscore_exit: Z-score threshold to exit a trade (default from settings).
            zscore_stop: Z-score threshold for stop loss (default from settings).
            lookback: Rolling window for spread z-score calculation (default from settings).
            coint_pvalue: Maximum p-value for cointegration test (default from settings).
        """
        self.zscore_entry = zscore_entry if zscore_entry is not None else getattr(settings, "PAIRS_ZSCORE_ENTRY", 2.0)
        self.zscore_exit = zscore_exit if zscore_exit is not None else getattr(settings, "PAIRS_ZSCORE_EXIT", 0.5)
        self.zscore_stop = zscore_stop if zscore_stop is not None else getattr(settings, "PAIRS_ZSCORE_STOP", 3.0)
        self.lookback = lookback if lookback is not None else getattr(settings, "PAIRS_LOOKBACK", 60)
        self.coint_pvalue = coint_pvalue if coint_pvalue is not None else getattr(settings, "PAIRS_COINT_PVALUE", 0.05)

    # ------------------------------------------------------------------
    # Cointegration Testing
    # ------------------------------------------------------------------

    def find_cointegrated_pairs(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.Series],
    ) -> List[Tuple[str, str, float, float, float]]:
        """Find cointegrated pairs from a list of symbols.

        Tests all possible pairs using the Engle-Granger cointegration test.
        Returns pairs sorted by p-value (most cointegrated first).

        Args:
            symbols: List of ticker symbols to test.
            data_dict: Dict mapping symbol → price Series (e.g. Close prices).

        Returns:
            List of tuples: (symbol_a, symbol_b, p_value, correlation, hedge_ratio)
            Only pairs with p_value < coint_pvalue are returned.
        """
        results = []
        for sym_a, sym_b in combinations(symbols, 2):
            price_a = data_dict.get(sym_a)
            price_b = data_dict.get(sym_b)
            if price_a is None or price_b is None:
                continue

            # Align the two series on shared index
            aligned = pd.concat([price_a, price_b], axis=1, join="inner").dropna()
            if len(aligned) < self.lookback * 2:
                logger.debug("Skipping %s/%s: insufficient data (%d rows)", sym_a, sym_b, len(aligned))
                continue

            col_a, col_b = aligned.iloc[:, 0], aligned.iloc[:, 1]

            try:
                _, p_value, _ = coint(col_a, col_b)
            except Exception as exc:
                logger.warning("Cointegration test failed for %s/%s: %s", sym_a, sym_b, exc)
                continue

            if p_value < self.coint_pvalue:
                correlation = float(col_a.corr(col_b))
                hedge_ratio = self.calculate_hedge_ratio(col_a, col_b)
                results.append((sym_a, sym_b, float(p_value), correlation, hedge_ratio))
                logger.info(
                    "Cointegrated pair found: %s/%s p=%.4f corr=%.3f hedge=%.4f",
                    sym_a, sym_b, p_value, correlation, hedge_ratio,
                )

        results.sort(key=lambda x: x[2])  # sort by p-value ascending
        logger.info("Found %d cointegrated pairs (p < %.2f)", len(results), self.coint_pvalue)
        return results

    # ------------------------------------------------------------------
    # Spread & Hedge Ratio
    # ------------------------------------------------------------------

    def calculate_hedge_ratio(self, price_a: pd.Series, price_b: pd.Series) -> float:
        """Calculate the OLS hedge ratio between two price series.

        Regresses price_a on price_b to find beta: price_a = alpha + beta * price_b.

        Args:
            price_a: Price series for stock A.
            price_b: Price series for stock B.

        Returns:
            OLS beta (hedge ratio). Returns 1.0 on failure.
        """
        try:
            X = add_constant(price_b.values)
            model = OLS(price_a.values, X).fit()
            return float(model.params[1])
        except Exception as exc:
            logger.warning("OLS hedge ratio calculation failed: %s", exc)
            return 1.0

    def calculate_rolling_hedge_ratio(
        self, price_a: pd.Series, price_b: pd.Series, window: int = None
    ) -> pd.Series:
        """Calculate a rolling OLS hedge ratio.

        For each window, regresses price_a on price_b and returns the beta.

        Args:
            price_a: Price series for stock A.
            price_b: Price series for stock B.
            window: Rolling window size (defaults to self.lookback).

        Returns:
            Series of rolling hedge ratios with same index as inputs.
        """
        window = window or self.lookback
        aligned = pd.concat([price_a, price_b], axis=1, join="inner").dropna()
        if len(aligned) < window:
            return pd.Series(1.0, index=aligned.index)

        col_a = aligned.iloc[:, 0]
        col_b = aligned.iloc[:, 1]
        hedge_ratios = pd.Series(np.nan, index=aligned.index)

        for i in range(window - 1, len(aligned)):
            window_a = col_a.iloc[i - window + 1: i + 1].values
            window_b = col_b.iloc[i - window + 1: i + 1].values
            try:
                X = add_constant(window_b)
                model = OLS(window_a, X).fit()
                hedge_ratios.iloc[i] = float(model.params[1])
            except Exception:
                hedge_ratios.iloc[i] = 1.0

        return hedge_ratios.ffill().fillna(1.0)

    def calculate_spread(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        hedge_ratio: float,
        window: int = None,
    ) -> pd.DataFrame:
        """Calculate the spread and its z-score between two price series.

        Spread = price_a - hedge_ratio * price_b
        Z-score = (spread - rolling_mean) / rolling_std

        Args:
            price_a: Price series for stock A.
            price_b: Price series for stock B.
            hedge_ratio: Static hedge ratio (or use calculate_rolling_hedge_ratio).
            window: Rolling window for z-score (defaults to self.lookback).

        Returns:
            DataFrame with columns: spread, spread_mean, spread_std, spread_zscore.
        """
        window = window or self.lookback
        aligned = pd.concat([price_a, price_b], axis=1, join="inner").dropna()
        col_a = aligned.iloc[:, 0]
        col_b = aligned.iloc[:, 1]

        spread = col_a - hedge_ratio * col_b
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        spread_zscore = (spread - spread_mean) / spread_std.replace(0, np.nan)

        return pd.DataFrame(
            {
                "spread": spread,
                "spread_mean": spread_mean,
                "spread_std": spread_std,
                "spread_zscore": spread_zscore,
            },
            index=aligned.index,
        )

    # ------------------------------------------------------------------
    # Signal Generation
    # ------------------------------------------------------------------

    def generate_signals(self, spread_zscore: pd.Series) -> pd.DataFrame:
        """Generate trading signals from the spread z-score.

        Signal rules:
          - Z-score < -ENTRY: BUY spread (buy A, sell B)
          - Z-score > +ENTRY: SELL spread (sell A, buy B)
          - |Z-score| < EXIT (after entry): CLOSE position
          - |Z-score| > STOP: STOP LOSS

        Args:
            spread_zscore: Series of z-score values for the spread.

        Returns:
            DataFrame with columns: signal, spread_zscore, entry_zscore.
        """
        signals = pd.DataFrame(index=spread_zscore.index)
        signals["spread_zscore"] = spread_zscore
        signals["signal"] = self.SIGNAL_NONE
        signals["entry_zscore"] = np.nan

        in_position = False
        position_side = 0  # +1 for long spread, -1 for short spread
        entry_zscore = np.nan

        for i, (idx, zscore) in enumerate(spread_zscore.items()):
            if pd.isna(zscore):
                continue

            if not in_position:
                if zscore < -self.zscore_entry:
                    signals.at[idx, "signal"] = self.SIGNAL_BUY
                    signals.at[idx, "entry_zscore"] = zscore
                    in_position = True
                    position_side = 1
                    entry_zscore = zscore
                elif zscore > self.zscore_entry:
                    signals.at[idx, "signal"] = self.SIGNAL_SELL
                    signals.at[idx, "entry_zscore"] = zscore
                    in_position = True
                    position_side = -1
                    entry_zscore = zscore
            else:
                # Check stop loss first
                if abs(zscore) > self.zscore_stop:
                    signals.at[idx, "signal"] = self.SIGNAL_STOP
                    signals.at[idx, "entry_zscore"] = entry_zscore
                    in_position = False
                    position_side = 0
                    entry_zscore = np.nan
                # Check mean reversion (close position)
                elif (position_side == 1 and zscore > -self.zscore_exit) or \
                     (position_side == -1 and zscore < self.zscore_exit):
                    signals.at[idx, "signal"] = self.SIGNAL_CLOSE
                    signals.at[idx, "entry_zscore"] = entry_zscore
                    in_position = False
                    position_side = 0
                    entry_zscore = np.nan

        return signals

    # ------------------------------------------------------------------
    # Order Execution Helper
    # ------------------------------------------------------------------

    def get_pair_orders(
        self,
        signal: int,
        symbol_a: str,
        symbol_b: str,
        hedge_ratio: float,
        capital_per_pair: float,
        price_a: float,
        price_b: float,
    ) -> List[Dict]:
        """Convert a spread signal into actual buy/sell orders for both legs.

        For a BUY spread: buy symbol_a, sell symbol_b.
        For a SELL spread: sell symbol_a, buy symbol_b.
        Quantities are sized to achieve equal dollar exposure based on hedge ratio.

        Args:
            signal: Signal value (SIGNAL_BUY, SIGNAL_SELL, SIGNAL_CLOSE, SIGNAL_STOP).
            symbol_a: First leg ticker symbol.
            symbol_b: Second leg ticker symbol.
            hedge_ratio: Hedge ratio (beta from OLS regression).
            capital_per_pair: Total capital to allocate to this pair.
            price_a: Current price of symbol_a.
            price_b: Current price of symbol_b.

        Returns:
            List of order dicts with keys: symbol, side, qty, reason.
            Empty list if signal is SIGNAL_NONE or prices are invalid.
        """
        if signal == self.SIGNAL_NONE or price_a <= 0 or price_b <= 0:
            return []

        if signal in (self.SIGNAL_CLOSE, self.SIGNAL_STOP):
            return [
                {"symbol": symbol_a, "side": "close", "qty": 0, "reason": "close_pair"},
                {"symbol": symbol_b, "side": "close", "qty": 0, "reason": "close_pair"},
            ]

        # Calculate quantities: allocate half capital to each leg
        half_capital = capital_per_pair / 2.0
        qty_a = max(1, int(half_capital / price_a))
        # B position is scaled by hedge ratio to maintain market neutrality
        qty_b = max(1, int(half_capital * hedge_ratio / price_b))

        if signal == self.SIGNAL_BUY:
            side_a, side_b = "buy", "sell"
        else:  # SIGNAL_SELL
            side_a, side_b = "sell", "buy"

        return [
            {"symbol": symbol_a, "side": side_a, "qty": qty_a, "reason": "pairs_trade"},
            {"symbol": symbol_b, "side": side_b, "qty": qty_b, "reason": "pairs_trade"},
        ]

    # ------------------------------------------------------------------
    # Performance Tracking
    # ------------------------------------------------------------------

    def get_pair_stats(self, trades: List[Dict]) -> Dict:
        """Calculate performance statistics for a pair's trade history.

        Args:
            trades: List of trade dicts, each with keys:
                    pnl, entry_date, exit_date, entry_zscore, exit_reason.

        Returns:
            Dict with return, sharpe, num_trades, win_rate.
        """
        if not trades:
            return {
                "num_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "avg_pnl": 0.0,
            }

        trades_df = pd.DataFrame(trades)
        pnls = trades_df["pnl"] if "pnl" in trades_df.columns else pd.Series(dtype=float)

        num_trades = len(pnls)
        total_pnl = float(pnls.sum())
        wins = pnls[pnls > 0]
        win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
        avg_pnl = float(pnls.mean()) if num_trades > 0 else 0.0

        std = float(pnls.std()) if num_trades > 1 else 0.0
        sharpe = (avg_pnl / std * np.sqrt(252)) if std > 0 else 0.0

        return {
            "num_trades": num_trades,
            "total_pnl": round(total_pnl, 4),
            "win_rate": round(win_rate, 4),
            "sharpe_ratio": round(sharpe, 4),
            "avg_pnl": round(avg_pnl, 4),
        }
