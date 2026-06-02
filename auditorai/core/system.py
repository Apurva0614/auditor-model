from __future__ import annotations
"""
AuditorSystem: end-to-end orchestration using the adapter interface.

Works with ANY model via the adapter interface — sklearn, PyTorch,
HuggingFace, API-based, or custom.
"""

import os

import numpy as np

from auditorai.core.auditor import AuditorModel
from auditorai.core.router import Router
from auditorai.utils.logging import get_logger

logger = get_logger(__name__)


class AuditorSystem:
    """
    The complete human-AI auditor system.

    Works with ANY model via the adapter interface:

      # sklearn
      from sklearn.ensemble import GradientBoostingClassifier
      from auditorai import AuditorSystem, wrap

      model = GradientBoostingClassifier().fit(X_train, y_train)
      system = AuditorSystem(wrap(model))
      system.train(X_val, y_val)
      result = system.predict(X_test)

      # PyTorch
      system = AuditorSystem(wrap(torch_model, adapter_type="pytorch",
                                  n_classes=3))

      # OpenAI
      system = AuditorSystem(wrap("gpt-4o", adapter_type="openai",
                                  api_key="sk-...",
                                  parse_response=my_parser,
                                  n_classes=2))

      # HuggingFace pipeline
      pipe = pipeline("text-classification", model="distilbert-...")
      system = AuditorSystem(wrap(pipe, adapter_type="huggingface"))

    Args:
      adapter: A ModelAdapter wrapping your primary model.
      auditor_threshold: Starting suppression threshold. Default 0.5.
    """

    def __init__(self, adapter: object,
                 auditor_threshold: float = 0.5):
        self.adapter = adapter
        # Keep self.primary_ as alias for backward compatibility with evaluate.py
        self.primary_ = adapter
        self.auditor_ = AuditorModel(auditor_threshold)
        self.router_: Router | None = None

    def train(self, X_val, y_val) -> None:
        """
        Trains the auditor on held-out validation data.

        CRITICAL: X_val must be data your primary model has NOT trained on.
        The auditor learns the primary model's real-world failure modes.
        Passing training data here will produce an unreliable auditor.

        Args:
          X_val: Validation features (same format your model expects).
          y_val: True labels for validation set.
        """
        logger.info(
            "Training auditor on held-out validation set (%d samples)...", len(X_val)
        )
        self.auditor_.fit(X_val, y_val, self.adapter)
        self.router_ = Router(self.adapter, self.auditor_,
                              self.auditor_.threshold)
        logger.info("AuditorSystem training complete.")

    def predict(self, X) -> dict:
        """
        Routes predictions through the auditor.
        Returns dict with keys:
          show_mask, suppress_mask, p_wrong, ai_predictions
        """
        if self.router_ is None:
            raise RuntimeError(
                "AuditorSystem has not been trained. Call train() first."
            )
        return self.router_.route(X)

    def auto_tune(self, X_val, y_val,
                  human_accuracy: float = 0.72) -> float:
        """Finds optimal threshold. Returns best tau."""
        best_tau = self.router_.best_threshold(X_val, y_val, human_accuracy)
        self.router_.set_threshold(best_tau)
        self.auditor_.threshold = best_tau
        logger.info("Auto-tuned threshold set to %.4f", best_tau)
        return best_tau

    def evaluate(self, X, y_true,
                 human_accuracy: float = 0.72) -> dict:
        """Compute comprehensive evaluation metrics.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y_true: True labels of shape (n_samples,).
            human_accuracy: Simulated human accuracy on suppressed cases.
                Defaults to 0.72.

        Returns:
            A dict with keys:
                - "ai_only_accuracy" (float)
                - "joint_accuracy" (float)
                - "accuracy_gain" (float)
                - "suppression_rate" (float)
                - "n_shown" (int)
                - "n_suppressed" (int)
                - "auditor_auroc" (float)
                - "auditor_precision" (float)
                - "auditor_recall" (float)
        """
        if self.router_ is None:
            raise RuntimeError(
                "AuditorSystem has not been trained. Call train() first."
            )
        result = self.router_.route(X)
        ai_predictions = result["ai_predictions"]
        suppress_mask = result["suppress_mask"]
        show_mask = result["show_mask"]

        ai_only_accuracy = float(np.mean(ai_predictions == y_true))

        n_shown = int(np.sum(show_mask))
        n_suppressed = int(np.sum(suppress_mask))
        suppression_rate = n_suppressed / len(y_true)

        if n_shown > 0:
            ai_acc_on_shown = float(
                np.mean(ai_predictions[show_mask] == y_true[show_mask])
            )
        else:
            ai_acc_on_shown = 0.0

        joint_accuracy = (
            ai_acc_on_shown * (1 - suppression_rate)
            + human_accuracy * suppression_rate
        )
        accuracy_gain = joint_accuracy - ai_only_accuracy

        auditor_metrics = self.auditor_.evaluate(X, y_true, self.adapter)

        return {
            "ai_only_accuracy": ai_only_accuracy,
            "joint_accuracy": joint_accuracy,
            "accuracy_gain": accuracy_gain,
            "suppression_rate": suppression_rate,
            "n_shown": n_shown,
            "n_suppressed": n_suppressed,
            "auditor_auroc": auditor_metrics["auroc"],
            "auditor_precision": auditor_metrics["precision"],
            "auditor_recall": auditor_metrics["recall"],
        }

    def save(self, directory: str) -> None:
        """Saves auditor_ to directory/auditor.joblib"""
        os.makedirs(directory, exist_ok=True)
        self.auditor_.save(os.path.join(directory, "auditor.joblib"))
        logger.info("AuditorSystem saved to %s", directory)

    def load(self, directory: str) -> None:
        """Loads auditor_ from directory. Adapter must be re-attached."""
        self.auditor_.load(os.path.join(directory, "auditor.joblib"))
        self.router_ = Router(
            self.adapter,
            self.auditor_,
            self.auditor_.threshold,
        )
        logger.info("AuditorSystem loaded from %s", directory)


def audit(adapter: object,
          X_val, y_val,
          X_test=None, y_test=None,
          human_accuracy: float = 0.72,
          auto_tune: bool = True) -> "AuditorSystem":
    """
    One-line entry point. Trains and optionally evaluates an auditor.

    Usage:
      system = audit(wrap(my_model), X_val, y_val, X_test, y_test)

    Args:
        adapter: A ModelAdapter wrapping your primary model.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Optional test features for evaluation.
        y_test: Optional test labels for evaluation.
        human_accuracy: Simulated human accuracy. Default 0.72.
        auto_tune: Whether to auto-tune the threshold. Default True.

    Returns a trained AuditorSystem. If X_test/y_test are provided,
    evaluation metrics are printed.
    """
    system = AuditorSystem(adapter)
    system.train(X_val, y_val)

    if auto_tune:
        system.auto_tune(X_val, y_val, human_accuracy=human_accuracy)

    if X_test is not None and y_test is not None:
        metrics = system.evaluate(X_test, y_test, human_accuracy=human_accuracy)
        logger.info("Evaluation metrics: %s", metrics)

    return system
