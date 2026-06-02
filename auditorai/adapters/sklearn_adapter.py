"""
Sklearn adapter for AuditorAI.

Wraps any scikit-learn compatible model (including XGBoost, LightGBM,
CatBoost with sklearn API, and sklearn Pipelines) into the unified
ModelAdapter interface.
"""

import numpy as np

from auditorai.adapters.base import ModelAdapter


class SklearnAdapter(ModelAdapter):
    """
    Wraps any scikit-learn compatible model.

    Supports:
      - Any model with .predict() and .predict_proba()
      - Models without predict_proba are wrapped automatically
        with CalibratedClassifierCV(cv=3)
      - sklearn Pipelines
      - XGBoost, LightGBM, CatBoost (sklearn API)

    Usage:
      from sklearn.ensemble import RandomForestClassifier
      from auditorai import AuditorSystem, wrap

      rf = RandomForestClassifier().fit(X_train, y_train)
      adapter = wrap(rf)  # or SklearnAdapter(rf)
      system = AuditorSystem(adapter)
      system.train(X_val, y_val)

    Args:
      model: Any fitted sklearn-compatible estimator.
      calibrate: If True and model lacks predict_proba, wraps with
                 CalibratedClassifierCV. Default True.
      feature_names: Optional list of feature names for logging.
    """

    def __init__(self, model, calibrate: bool = True,
                 feature_names: list = None):
        self.model = model
        self.calibrate = calibrate
        self.feature_names = feature_names
        self._calibrated_model = None

        # If model doesn't have predict_proba, attempt calibration
        if not hasattr(model, "predict_proba") and calibrate:
            self._needs_calibration = True
        else:
            self._needs_calibration = False

    def predict(self, X) -> np.ndarray:
        """Returns predicted class labels."""
        model = self._calibrated_model if self._calibrated_model is not None else self.model
        return model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """
        Returns calibrated class probabilities.
        If model was wrapped with CalibratedClassifierCV, returns
        those probabilities. Otherwise calls model.predict_proba directly.
        """
        model = self._calibrated_model if self._calibrated_model is not None else self.model

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)
        else:
            # Fallback: build probas from decision_function if available
            if hasattr(model, "decision_function"):
                decision = model.decision_function(X)
                # Sigmoid to get rough probabilities
                from scipy.special import expit
                if decision.ndim == 1:
                    p1 = expit(decision)
                    probas = np.column_stack([1 - p1, p1])
                else:
                    # Multi-class: softmax
                    exp_d = np.exp(decision - decision.max(axis=1, keepdims=True))
                    probas = exp_d / exp_d.sum(axis=1, keepdims=True)
            else:
                raise AttributeError(
                    f"{type(self.model).__name__} has no predict_proba or "
                    f"decision_function. Set calibrate=True and call "
                    f"adapter.fit(X, y) to calibrate."
                )

        self.validate_probas(probas)
        return probas

    def fit(self, X, y):
        """
        Fits the underlying model if not already fitted.
        If the model lacks predict_proba and calibrate=True,
        wraps it with CalibratedClassifierCV.
        """
        from sklearn.calibration import CalibratedClassifierCV

        if self._needs_calibration:
            calibrated = CalibratedClassifierCV(self.model, cv=3)
            calibrated.fit(X, y)
            self._calibrated_model = calibrated
            self._needs_calibration = False
        else:
            self.model.fit(X, y)
        return self
