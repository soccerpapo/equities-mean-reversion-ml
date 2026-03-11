"""Adaptive Strategy: Regime-Based Strategy Switching.

This module implements an adaptive trading strategy that:
  1. Detects the current market regime using the RegimeDetector
  2. Selects the appropriate strategy for that regime
  3. Manages smooth transitions between strategies over TRANSITION_DAYS
  4. Allocates capital according to REGIME_ALLOCATIONS

Regime mapping:
  Regime 0 (Low vol / Mean-reverting)  → Pairs Trading
  Regime 1 (Normal / Trending)         → Momentum
  Regime 2 (High vol / Crisis)         → Cash (no trading)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from strategy.pairs_trading import PairsTrader
from strategy.momentum import MomentumTrader
from strategy.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


class AdaptiveTrader:
    """Regime-based adaptive strategy that switches between Pairs and Momentum.

    Capital allocation by regime (configurable via REGIME_ALLOCATIONS):
      Regime 0: 70% pairs trading, 20% momentum, 10% cash
      Regime 1: 20% pairs trading, 70% momentum, 10% cash
      Regime 2: 0% trading, 100% cash
    """

    def __init__(
        self,
        regime_allocations: Dict = None,
        transition_days: int = None,
        n_regime_components: int = None,
    ):
        """Initialise the AdaptiveTrader.

        Args:
            regime_allocations: Dict mapping regime → {pairs, momentum, cash} fractions.
                                 Defaults to settings.REGIME_ALLOCATIONS.
            transition_days: Number of days over which to phase transitions.
                             Defaults to settings.TRANSITION_DAYS.
            n_regime_components: Number of GMM components for regime detection.
                                  Defaults to settings.REGIME_N_COMPONENTS.
        """
        self.regime_allocations = regime_allocations or getattr(
            settings, "REGIME_ALLOCATIONS",
            {
                0: {"pairs": 0.7, "momentum": 0.2, "cash": 0.1},
                1: {"pairs": 0.2, "momentum": 0.7, "cash": 0.1},
                2: {"pairs": 0.0, "momentum": 0.0, "cash": 1.0},
            },
        )
        self.transition_days = transition_days if transition_days is not None else getattr(settings, "TRANSITION_DAYS", 3)
        n_components = n_regime_components if n_regime_components is not None else getattr(settings, "REGIME_N_COMPONENTS", 3)

        self.pairs_trader = PairsTrader()
        self.momentum_trader = MomentumTrader()
        self.regime_detector = RegimeDetector(n_components=n_components)

        self._current_regime: Optional[int] = None
        self._transition_counter: int = 0
        self._pending_regime: Optional[int] = None

    # ------------------------------------------------------------------
    # Regime → Strategy Selection
    # ------------------------------------------------------------------

    def select_strategy(self, regime: int):
        """Return the active strategy object for the given regime.

        Args:
            regime: Regime label (0=low-vol, 1=normal/trending, 2=crisis).

        Returns:
            PairsTrader for regime 0, MomentumTrader for regime 1, None for regime 2.
        """
        if regime == 0:
            return self.pairs_trader
        elif regime == 1:
            return self.momentum_trader
        else:  # regime 2 or unknown → cash
            return None

    # ------------------------------------------------------------------
    # Capital Allocation
    # ------------------------------------------------------------------

    def allocate_capital(self, regime: int, total_capital: float) -> Dict[str, float]:
        """Allocate capital across strategies based on regime.

        Args:
            regime: Current regime label (0, 1, or 2).
            total_capital: Total portfolio value to allocate.

        Returns:
            Dict with keys 'pairs', 'momentum', 'cash' and dollar amounts.
        """
        alloc_pcts = self.regime_allocations.get(regime, {"pairs": 0.0, "momentum": 0.0, "cash": 1.0})
        return {
            "pairs": total_capital * alloc_pcts.get("pairs", 0.0),
            "momentum": total_capital * alloc_pcts.get("momentum", 0.0),
            "cash": total_capital * alloc_pcts.get("cash", 1.0),
        }

    # ------------------------------------------------------------------
    # Smooth Transitions
    # ------------------------------------------------------------------

    def manage_transition(
        self,
        old_regime: int,
        new_regime: int,
        current_positions: Dict,
    ) -> List[Dict]:
        """Manage a gradual regime transition over TRANSITION_DAYS days.

        On each call during a transition, reduces old strategy positions
        by 1/TRANSITION_DAYS and signals the caller to build new positions
        at the same fraction.

        Args:
            old_regime: Previous regime label.
            new_regime: New regime label.
            current_positions: Dict mapping symbol → position dict with at least
                               {'qty': int, 'strategy': str}.

        Returns:
            List of order dicts with keys: symbol, side, qty, reason.
            The caller should execute these orders and then build new positions.
        """
        if not current_positions:
            return []

        orders = []
        # Reduce 1/TRANSITION_DAYS of existing positions each call
        reduction_fraction = 1.0 / max(1, self.transition_days)

        for symbol, pos in current_positions.items():
            qty = int(pos.get("qty", 0))
            side = pos.get("side", "long")
            if qty <= 0:
                continue

            reduce_qty = max(1, int(qty * reduction_fraction))
            close_side = "sell" if side == "long" else "buy"
            orders.append(
                {
                    "symbol": symbol,
                    "side": close_side,
                    "qty": reduce_qty,
                    "reason": f"regime_transition_{old_regime}_to_{new_regime}",
                }
            )

        logger.info(
            "Regime transition %d → %d: reducing %d positions by %.0f%%",
            old_regime, new_regime, len(orders), reduction_fraction * 100,
        )
        return orders

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        total_capital: float = 100_000.0,
    ) -> Dict:
        """Run the adaptive strategy for the current period.

        Detects regime, selects strategy, generates signals, and returns
        a combined order set with capital allocation information.

        Args:
            symbols: List of ticker symbols to trade.
            data_dict: Dict mapping symbol → OHLCV DataFrame.
            total_capital: Total portfolio capital.

        Returns:
            Dict with keys:
              regime: current regime (0, 1, or 2)
              strategy: 'pairs', 'momentum', or 'cash'
              allocation: capital allocation dict
              signals: per-symbol signals DataFrame (for momentum / single-asset)
              pairs: list of cointegrated pairs (for pairs trading)
              orders: list of suggested order dicts
        """
        # Detect regime using any available symbol (prefer SPY if present)
        ref_symbol = "SPY" if "SPY" in data_dict else symbols[0] if symbols else None
        if ref_symbol is None or ref_symbol not in data_dict:
            logger.warning("No reference symbol found for regime detection.")
            regime = 1
            confidence = 0.5
        else:
            ref_df = data_dict[ref_symbol]
            self.regime_detector.fit(ref_df)
            regime, confidence = self.regime_detector.detect_regime(ref_df)

        logger.info("AdaptiveTrader: regime=%d (confidence=%.2f)", regime, confidence)

        # Detect regime change and manage transitions
        if self._current_regime is None:
            # First call — set initial regime directly
            self._current_regime = regime
        elif regime != self._current_regime and self._pending_regime is None:
            # New regime detected — start transition
            logger.info(
                "Regime change detected: %d → %d", self._current_regime, regime
            )
            self._pending_regime = regime
            self._transition_counter = self.transition_days
        elif self._pending_regime is not None and self._transition_counter > 0:
            # Transition in progress — count down
            self._transition_counter -= 1
            if self._transition_counter == 0:
                self._current_regime = self._pending_regime
                self._pending_regime = None
        # When a transition is in progress, keep _current_regime unchanged
        # until the counter reaches zero; do NOT overwrite it unconditionally.

        # Use the stable _current_regime for allocation and strategy selection so
        # that capital isn't reallocated mid-transition while positions from the
        # old regime are still being unwound.
        active_regime = self._current_regime if self._current_regime is not None else regime
        allocation = self.allocate_capital(active_regime, total_capital)
        strategy = self.select_strategy(active_regime)
        strategy_name = {0: "pairs", 1: "momentum", 2: "cash"}.get(active_regime, "cash")

        signals: Dict[str, pd.DataFrame] = {}
        pairs: List = []
        orders: List[Dict] = []

        if active_regime == 2 or strategy is None:
            logger.info("Regime 2 (crisis): holding cash, no new orders.")
        elif active_regime == 0 and isinstance(strategy, PairsTrader):
            # Pairs trading
            price_series = {sym: df["Close"] for sym, df in data_dict.items() if "Close" in df.columns}
            pairs = strategy.find_cointegrated_pairs(symbols, price_series)
            capital_per_pair = allocation["pairs"] / max(1, len(pairs))
            for sym_a, sym_b, p_val, corr, hedge in pairs:
                if sym_a not in data_dict or sym_b not in data_dict:
                    continue
                spread_df = strategy.calculate_spread(
                    data_dict[sym_a]["Close"],
                    data_dict[sym_b]["Close"],
                    hedge,
                )
                if spread_df.empty or spread_df["spread_zscore"].dropna().empty:
                    continue
                pair_signals = strategy.generate_signals(spread_df["spread_zscore"])
                latest_signal = int(pair_signals["signal"].iloc[-1])
                latest_price_a = float(data_dict[sym_a]["Close"].iloc[-1])
                latest_price_b = float(data_dict[sym_b]["Close"].iloc[-1])
                pair_orders = strategy.get_pair_orders(
                    latest_signal, sym_a, sym_b, hedge,
                    capital_per_pair, latest_price_a, latest_price_b,
                )
                orders.extend(pair_orders)
                signals[f"{sym_a}/{sym_b}"] = pair_signals
        elif active_regime == 1 and isinstance(strategy, MomentumTrader):
            # Momentum trading
            top_symbols, _ = strategy.rank_and_select(data_dict, top_n=strategy.top_n)
            for symbol in top_symbols:
                if symbol not in data_dict:
                    continue
                sym_signals = strategy.generate_signals(data_dict[symbol])
                signals[symbol] = sym_signals
                latest = sym_signals.iloc[-1]
                sig = int(latest.get("signal", 0))
                strength = float(latest.get("signal_strength", 0))
                if sig != 0 and strength > 0:
                    price = float(data_dict[symbol]["Close"].iloc[-1])
                    qty = max(1, int(allocation["momentum"] / len(top_symbols) / price))
                    orders.append(
                        {
                            "symbol": symbol,
                            "side": "buy" if sig == 1 else "sell",
                            "qty": qty,
                            "reason": "momentum_signal",
                        }
                    )

        return {
            "regime": regime,
            "regime_confidence": confidence,
            "strategy": strategy_name,
            "allocation": allocation,
            "signals": signals,
            "pairs": pairs,
            "orders": orders,
        }
