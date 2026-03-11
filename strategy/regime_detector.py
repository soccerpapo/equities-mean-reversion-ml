"""Market regime detector using Gaussian Mixture Model for volatility clustering.

Detects 3 regimes:
  Regime 0: Low volatility / Mean-reverting  → TRADE mean reversion (full size)
  Regime 1: Normal / Trending                → REDUCED position sizes (50%)
  Regime 2: High volatility / Crisis         → NO TRADING (sit out)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detects market regimes using a Gaussian Mixture Model.

    Features used:
      - 20-day realized volatility
      - Volatility of volatility (std of rolling vol)
      - Average absolute daily return
      - Volume ratio (current vs 20-day average)
      - Autocorrelation of returns (negative = mean reverting)
      - Kurtosis of returns (dispersion)

    Regimes (sorted by volatility level after fitting):
      0: Low-vol / mean-reverting  → position multiplier 1.0
      1: Normal / trending         → position multiplier 0.5
      2: High-vol / crisis         → position multiplier 0.0
    """

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._gmm: GaussianMixture | None = None
        self._scaler = StandardScaler()
        self._sorted_gmm_labels: list[int] | None = None  # _sorted_gmm_labels[i] = GMM label ranked i-th by vol
        self._is_fitted = False
        # Regime persistence filter state
        self._confirmed_regime: int | None = None   # last regime that passed persistence check
        self._candidate_regime: int | None = None   # regime being counted toward persistence
        self._candidate_count: int = 0              # consecutive days seen for candidate_regime
        self._persistence_threshold: int = 5        # days required before confirming a switch
        self._confidence_threshold: float = 0.7     # minimum confidence to consider a regime change

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """Fit the GMM on historical data.

        Args:
            df: OHLCV DataFrame (Close and Volume required).

        Returns:
            self (for chaining)
        """
        features = self._extract_features(df)
        if features is None or features.shape[0] < self.n_components * 10:
            logger.warning("Insufficient data for regime detection; using default model.")
            self._is_fitted = False
            return self

        X_scaled = self._scaler.fit_transform(features)
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=self.random_state,
            n_init=5,
        )
        self._gmm.fit(X_scaled)

        # Determine regime ordering by average realized volatility of each component.
        # Component with lowest vol → regime 0; highest → regime 2.
        # _sorted_gmm_labels[i] = the original GMM component index that ranks i-th
        # by volatility. Used to map raw GMM labels to standardised regimes.
        labels = self._gmm.predict(X_scaled)
        vol_col = features[:, 0]  # first feature is 20-day realized vol
        mean_vol = [vol_col[labels == k].mean() for k in range(self.n_components)]
        self._sorted_gmm_labels = list(np.argsort(mean_vol))

        self._is_fitted = True
        # Reset persistence filter state when model is re-fitted
        self._confirmed_regime = None
        self._candidate_regime = None
        self._candidate_count = 0
        logger.info(
            "RegimeDetector fitted. Vol means per regime (sorted low→high): %s",
            [round(mean_vol[k], 5) for k in self._sorted_gmm_labels],
        )
        return self

    def detect_regime(self, df: pd.DataFrame) -> tuple[int, float]:
        """Detect the current market regime with persistence and confidence filtering.

        Applies two layers of filtering to reduce noisy regime switches:
          1. Confidence threshold: only consider a regime change when the GMM
             posterior probability for the new regime exceeds 0.7.
          2. Persistence filter: require the same regime to be detected for at
             least 5 consecutive calls before confirming the switch.

        Args:
            df: OHLCV DataFrame (requires at least 30 rows for feature calculation).

        Returns:
            Tuple of (regime_label, confidence) where regime_label is 0, 1, or 2
            and confidence is the posterior probability for the raw detected regime.
            Falls back to regime 1 (normal) if model is not fitted.
        """
        if not self._is_fitted or self._gmm is None:
            return 1, 0.5

        features = self._extract_features(df)
        if features is None or len(features) == 0:
            return 1, 0.5

        # Use only the last observation
        last_features = features[[-1], :]
        X_scaled = self._scaler.transform(last_features)
        raw_label = int(self._gmm.predict(X_scaled)[0])
        proba = self._gmm.predict_proba(X_scaled)[0]
        confidence = float(proba[raw_label])

        # Map raw GMM label to standardised regime (0=low-vol, 1=normal, 2=high-vol)
        raw_regime = self._sorted_gmm_labels.index(raw_label)

        # Apply confidence threshold: if confidence is too low, keep the current
        # confirmed regime (or default to normal if no regime confirmed yet).
        if confidence < self._confidence_threshold:
            if self._confirmed_regime is not None:
                return self._confirmed_regime, confidence
            return 1, confidence

        # Apply persistence filter: only confirm a new regime after seeing it
        # for at least _persistence_threshold consecutive calls.
        if raw_regime == self._candidate_regime:
            self._candidate_count += 1
        else:
            self._candidate_regime = raw_regime
            self._candidate_count = 1

        if self._candidate_count >= self._persistence_threshold:
            self._confirmed_regime = self._candidate_regime

        # Return the confirmed regime (or default until enough data accumulated).
        if self._confirmed_regime is not None:
            return self._confirmed_regime, confidence
        return 1, confidence

    def get_position_multiplier(self, regime: int) -> float:
        """Return the position size multiplier for a given regime.

        Args:
            regime: Regime label (0, 1, or 2)

        Returns:
            1.0 for regime 0, 0.5 for regime 1, 0.0 for regime 2
        """
        mapping = {0: 1.0, 1: 0.5, 2: 0.0}
        return mapping.get(regime, 0.5)

    def save_model(self, path: str) -> None:
        """Persist the fitted model to disk.

        Args:
            path: File path (e.g. 'models/regime_model.joblib')
        """
        if not self._is_fitted:
            logger.warning("Model is not fitted; nothing saved.")
            return
        joblib.dump({"gmm": self._gmm, "scaler": self._scaler, "sorted_gmm_labels": self._sorted_gmm_labels}, path)
        logger.info("RegimeDetector saved to %s", path)

    @classmethod
    def load_model(cls, path: str) -> "RegimeDetector":
        """Load a previously fitted model from disk.

        Args:
            path: File path to the saved model

        Returns:
            A fitted RegimeDetector instance
        """
        data = joblib.load(path)
        detector = cls()
        detector._gmm = data["gmm"]
        detector._scaler = data["scaler"]
        detector._sorted_gmm_labels = data["sorted_gmm_labels"]
        detector._is_fitted = True
        return detector

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray | None:
        """Extract regime-detection features from a price DataFrame.

        Returns a 2-D numpy array of shape (n_valid_rows, 6), or None if
        there is not enough data.
        """
        close = df["Close"]
        returns = close.pct_change()

        # 1. 20-day realized volatility (annualised)
        vol_20 = returns.rolling(window=20).std() * np.sqrt(252)

        # 2. Volatility-of-volatility: 10-day std of the rolling vol
        vol_of_vol = vol_20.rolling(window=10).std()

        # 3. Average absolute daily return (20-day mean)
        abs_ret_20 = returns.abs().rolling(window=20).mean()

        # 4. Volume ratio: 5-day avg / 20-day avg
        if "Volume" in df.columns:
            vol_ratio = (
                df["Volume"].rolling(window=5).mean()
                / df["Volume"].rolling(window=20).mean().replace(0, np.nan)
            )
        else:
            vol_ratio = pd.Series(1.0, index=df.index)

        # 5. Autocorrelation of returns at lag-1 (negative → mean-reverting)
        autocorr = returns.rolling(window=20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
        )

        # 6. Kurtosis of 20-day returns
        kurtosis = returns.rolling(window=20).kurt()

        feature_df = pd.DataFrame(
            {
                "vol_20": vol_20,
                "vol_of_vol": vol_of_vol,
                "abs_ret_20": abs_ret_20,
                "vol_ratio": vol_ratio,
                "autocorr": autocorr,
                "kurtosis": kurtosis,
            }
        )

        feature_df = feature_df.dropna()
        if len(feature_df) < self.n_components * 5:
            return None

        return feature_df.values
