import logging
import os
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from config import settings

logger = logging.getLogger(__name__)


class MLSignalFilter:
    """ML-based signal filter using LightGBM."""

    FEATURE_COLS = [
        "zscore", "rsi", "macd", "macd_signal", "macd_hist",
        "bb_pct_b", "bb_bandwidth", "volume_zscore", "volatility",
        "roc", "atr",
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5", "return_lag_10",
        "skewness_20", "kurtosis_20", "vol_ratio", "volume_trend",
        "dist_sma50", "dist_sma200", "autocorr_10", "intraday_range", "gap",
        "day_of_week", "month",
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
        if "day_of_week" not in result.columns:
            result["day_of_week"] = result.index.dayofweek
        if "month" not in result.columns:
            result["month"] = result.index.month

        available = [c for c in self.FEATURE_COLS if c in result.columns]
        self._feature_cols = available
        return result[available].copy()

    def create_labels(self, df: pd.DataFrame, forward_returns_period: int = 5, signal_type: int = 1) -> pd.Series:
        """Create binary labels based on risk-adjusted forward returns.

        Labels use future returns, which is correct for supervised learning — features
        at time t are paired with the realized return from t to t+N. At inference time
        only past features are used, so there is no look-ahead bias in the live system.
        Rows with unknown future returns (end of series) become NaN and are dropped by
        the caller before training.

        Args:
            df: DataFrame with Close prices
            forward_returns_period: Number of days to look forward
            signal_type: 1 for BUY signals (label=1 if forward return > transaction cost),
                        -1 for SELL signals (label=1 if forward return is negative)

        Returns:
            Series of binary labels aligned to each row's date
        """
        transaction_cost = 0.0002  # 0.02%
        fwd_return = df["Close"].shift(-forward_returns_period) / df["Close"] - 1
        if signal_type == -1:
            return (fwd_return < 0).astype(int)
        return (fwd_return > transaction_cost).astype(int)

    def _tune_and_train(self, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
        """Find best hyperparameters using TimeSeriesSplit, then train on full training set.

        Args:
            X_train: Training features (DataFrame with column names)
            y_train: Training labels

        Returns:
            Best fitted LGBMClassifier
        """
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_child_samples": [10, 20, 50],
        }

        tscv = TimeSeriesSplit(n_splits=5)
        best_score = -np.inf
        best_params = {}

        scale_pos_weight_value = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        for n_est in param_grid["n_estimators"]:
            for depth in param_grid["max_depth"]:
                for lr in param_grid["learning_rate"]:
                    for min_child in param_grid["min_child_samples"]:
                        scores = []
                        model = lgb.LGBMClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            min_child_samples=min_child,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            scale_pos_weight=scale_pos_weight_value,
                            random_state=42,
                            verbose=-1,
                        )
                        for train_idx, val_idx in tscv.split(X_train):
                            X_tr = X_train.iloc[train_idx]
                            y_tr = y_train.iloc[train_idx]
                            X_val = X_train.iloc[val_idx]
                            y_val = y_train.iloc[val_idx]
                            model.fit(X_tr, y_tr)
                            scores.append(model.score(X_val, y_val))
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {
                                "n_estimators": n_est,
                                "max_depth": depth,
                                "learning_rate": lr,
                                "min_child_samples": min_child,
                            }

        logger.info(f"Best CV params: {best_params} (score={best_score:.4f})")
        print(f"Best hyperparameters: {best_params}")

        final_model = lgb.LGBMClassifier(
            **best_params,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight_value,
            random_state=42,
            verbose=-1,
        )
        final_model.fit(X_train, y_train)
        return final_model

    def train(self, df: pd.DataFrame) -> None:
        """Train LightGBM classifier with time-series split.

        Args:
            df: DataFrame with indicators and Close prices
        """
        forward_period = getattr(settings, "FORWARD_RETURN_PERIOD", 5)
        X = self.prepare_features(df)
        y = self.create_labels(df, forward_returns_period=forward_period)

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

        X_train_scaled = pd.DataFrame(
            self._scaler.fit_transform(X_train),
            columns=self._feature_cols,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self._scaler.transform(X_test),
            columns=self._feature_cols,
            index=X_test.index,
        )

        self._model = self._tune_and_train(X_train_scaled, y_train)

        y_pred = self._model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)
        logger.info(f"ML model classification report:\n{report}")
        print(report)

        self._log_feature_importance()

    def train_multi_symbol(self, symbols: List[str]) -> None:
        """Train on multiple symbols combined for richer training data.

        Fetches ML_LOOKBACK_YEARS years of data for each symbol, computes indicators,
        concatenates all data into one large training set, and trains the model.

        Args:
            symbols: List of ticker symbols to include in training
        """
        import time
        from data.fetcher import DataFetcher
        from features.indicators import IndicatorEngine

        lookback_years = getattr(settings, "ML_LOOKBACK_YEARS", 5)
        forward_period = getattr(settings, "FORWARD_RETURN_PERIOD", 5)
        period = f"{lookback_years}y"

        fetcher = DataFetcher()
        ind_engine = IndicatorEngine()

        all_X: List[pd.DataFrame] = []
        all_y: List[pd.Series] = []

        for symbol in symbols:
            logger.info(f"Fetching training data for {symbol}...")
            df = fetcher.fetch_historical(symbol, period=period)
            if df.empty:
                logger.warning(f"No data for {symbol}, skipping")
                continue
            df = ind_engine.compute_all(df)
            X = self.prepare_features(df)
            y = self.create_labels(df, forward_returns_period=forward_period)

            combined = pd.concat([X, y.rename("label")], axis=1).dropna()
            if len(combined) < 50:
                logger.warning(f"Not enough data for {symbol}, skipping")
                continue
            all_X.append(combined[self._feature_cols])
            all_y.append(combined["label"])
            time.sleep(0.2)

        if not all_X:
            logger.error("No training data collected across all symbols")
            return

        X_all = pd.concat(all_X, ignore_index=True)
        y_all = pd.concat(all_y, ignore_index=True)
        logger.info(f"Combined training set: {len(X_all)} samples from {len(all_X)} symbols")

        split_idx = int(len(X_all) * 0.8)
        X_train = X_all.iloc[:split_idx]
        X_test = X_all.iloc[split_idx:]
        y_train = y_all.iloc[:split_idx]
        y_test = y_all.iloc[split_idx:]

        X_train_scaled = pd.DataFrame(
            self._scaler.fit_transform(X_train),
            columns=self._feature_cols,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self._scaler.transform(X_test),
            columns=self._feature_cols,
            index=X_test.index,
        )

        self._model = self._tune_and_train(X_train_scaled, y_train)

        y_pred = self._model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)
        logger.info(f"Multi-symbol ML model classification report:\n{report}")
        print(report)

        self._log_feature_importance()

    def _log_feature_importance(self) -> None:
        """Log top 10 most important features."""
        if self._model is None:
            return
        importance = self.get_feature_importance()
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top10 = sorted_imp[:10]
        logger.info("Top 10 feature importances:")
        for feat, score in top10:
            logger.info(f"  {feat}: {score:.4f}")

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
        X_scaled = pd.DataFrame(
            self._scaler.transform(X_clean),
            columns=self._feature_cols,
            index=X_clean.index,
        )
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
