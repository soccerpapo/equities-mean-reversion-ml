import logging
import os
from typing import Dict, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from config import settings

logger = logging.getLogger(__name__)


class MLSignalFilter:
    """ML-based signal filter using LightGBM."""

    FEATURE_COLS = [
        "zscore", "rsi", "macd", "macd_signal", "macd_hist",
        "bb_pct_b", "bb_bandwidth", "volume_zscore", "volatility",
        "roc", "atr", "day_of_week", "month",
    ]

    def __init__(self):
        self._model: Optional[lgb.LGBMClassifier] = None
        self._scaler = StandardScaler()
        self._feature_cols: list = []

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix from indicators.

        Args:
            df: DataFrame with computed indicators

        Returns:
            DataFrame with features ready for ML
        """
        result = df.copy()
        result["day_of_week"] = result.index.dayofweek
        result["month"] = result.index.month

        available = [c for c in self.FEATURE_COLS if c in result.columns]
        self._feature_cols = available
        return result[available].copy()

    def create_labels(self, df: pd.DataFrame, forward_returns_period: int = 5) -> pd.Series:
        """Create binary labels: 1 if forward return > 0 over the next N days.

        Labels use future returns, which is correct for supervised learning — features
        at time t are paired with the realized return from t to t+N. At inference time
        only past features are used, so there is no look-ahead bias in the live system.
        Rows with unknown future returns (end of series) become NaN and are dropped by
        the caller before training.

        Args:
            df: DataFrame with Close prices
            forward_returns_period: Number of days to look forward

        Returns:
            Series of binary labels aligned to each row's date
        """
        fwd_return = df["Close"].shift(-forward_returns_period) / df["Close"] - 1
        return (fwd_return > 0).astype(int)

    def train(self, df: pd.DataFrame) -> None:
        """Train LightGBM classifier with time-series split.

        Args:
            df: DataFrame with indicators and Close prices
        """
        X = self.prepare_features(df)
        y = self.create_labels(df)

        combined = pd.concat([X, y.rename("label")], axis=1).dropna()
        X_clean = combined[self._feature_cols]
        y_clean = combined["label"]

        if len(X_clean) < 50:
            logger.warning("Not enough data to train")
            return

        split_idx = int(len(X_clean) * 0.8)
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        self._model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            random_state=42,
            verbose=-1,
        )
        self._model.fit(X_train_scaled, y_train)

        y_pred = self._model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)
        logger.info(f"ML model classification report:\n{report}")
        print(report)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict probability of profitable trade.

        Args:
            df: DataFrame with indicators

        Returns:
            Series of probabilities
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.prepare_features(df)
        X_clean = X.reindex(columns=self._feature_cols).fillna(0)
        X_scaled = self._scaler.transform(X_clean)
        proba = self._model.predict_proba(X_scaled)[:, 1]
        return pd.Series(proba, index=df.index)

    def filter_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only signals where ML confidence exceeds threshold.

        Args:
            df: DataFrame with 'signal' column

        Returns:
            DataFrame with filtered signals
        """
        if self._model is None:
            logger.warning("Model not trained; returning signals unfiltered")
            return df
        result = df.copy()
        proba = self.predict(df)
        result["ml_confidence"] = proba
        threshold = settings.ML_CONFIDENCE_THRESHOLD
        result.loc[
            (result["signal"] != 0) & (result["ml_confidence"] < threshold),
            "signal"
        ] = 0
        return result

    def save_model(self, path: str) -> None:
        """Save model and scaler to disk.

        Args:
            path: File path to save model
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({"model": self._model, "scaler": self._scaler, "features": self._feature_cols}, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model and scaler from disk.

        Args:
            path: File path to load model from
        """
        data = joblib.load(path)
        self._model = data["model"]
        self._scaler = data["scaler"]
        self._feature_cols = data["features"]
        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance dict.

        Returns:
            Dict mapping feature names to importance scores
        """
        if self._model is None:
            return {}
        importances = self._model.feature_importances_
        return dict(zip(self._feature_cols, importances))
