"""
Primary classification model with calibrated probability output.
"""

import os

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

logger = get_logger(__name__)


class PrimaryModel:
    """Calibrated classification model serving as the primary AI decision-maker.

    Wraps one of three sklearn classifiers inside a StandardScaler +
    CalibratedClassifierCV pipeline for reliable probability estimates.

    Args:
        model_type: One of "random_forest", "gradient_boosting", or "logistic".

    Raises:
        ValueError: If `model_type` is not one of the supported options.
    """

    SUPPORTED_TYPES = ("random_forest", "gradient_boosting", "logistic")

    def __init__(self, model_type: str = "random_forest") -> None:
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from: {self.SUPPORTED_TYPES}"
            )
        self.model_type = model_type
        self.pipeline_: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        """Build and return the scaler + calibrated classifier pipeline.

        Returns:
            A sklearn Pipeline with StandardScaler and CalibratedClassifierCV.
        """
        if self.model_type == "random_forest":
            base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "gradient_boosting":
            base_clf = GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
        else:  # logistic
            base_clf = LogisticRegression(max_iter=1000, random_state=42)

        calibrated = CalibratedClassifierCV(base_clf, cv=3)
        return Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the pipeline on training data.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).
        """
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        logger.info("PrimaryModel fitted on %d samples.", len(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            Predicted labels array of shape (n_samples,).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.pipeline_ is None:
            raise RuntimeError("PrimaryModel is not fitted. Call fit() first.")
        return self.pipeline_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            Probability array of shape (n_samples, n_classes).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.pipeline_ is None:
            raise RuntimeError("PrimaryModel is not fitted. Call fit() first.")
        return self.pipeline_.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the model on a dataset.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: True label array of shape (n_samples,).

        Returns:
            A dict with keys:
                - "accuracy" (float): Fraction of correct predictions.
                - "roc_auc" (float): ROC-AUC score.
                - "report" (str): sklearn classification report string.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.pipeline_ is None:
            raise RuntimeError("PrimaryModel is not fitted. Call fit() first.")
        preds = self.predict(X)
        probas = self.predict_proba(X)
        accuracy = float(np.mean(preds == y))
        n_classes = probas.shape[1]
        if n_classes == 2:
            roc_auc = roc_auc_score(y, probas[:, 1])
        else:
            roc_auc = roc_auc_score(y, probas, multi_class="ovr")
        report = classification_report(y, preds)
        return {"accuracy": accuracy, "roc_auc": float(roc_auc), "report": report}

    def save(self, path: str) -> None:
        """Serialize the fitted model to disk.

        Creates parent directories if they do not exist.

        Args:
            path: Destination file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        logger.info("PrimaryModel saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved PrimaryModel from disk.

        Updates this instance's __dict__ with the loaded object's state.

        Args:
            path: Path to the serialized model file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        logger.info("PrimaryModel loaded from %s", path)
