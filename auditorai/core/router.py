from __future__ import annotations
"""
Router: decides which samples the AI system should answer vs. suppress.
"""

import numpy as np
import pandas as pd

from auditorai.utils.logging import get_logger

logger = get_logger(__name__)


class Router:
    """Routes samples to the AI or suppresses them based on auditor scores.

    The router queries the primary model (via adapter) for predictions and the
    auditor for suppression decisions, then combines them into a routing result.

    Args:
        adapter: A fitted ModelAdapter instance.
        auditor_model: A fitted AuditorModel instance.
        threshold: Probability threshold; samples with P(wrong) >= threshold
            are suppressed. Defaults to 0.5.
    """

    def __init__(
        self,
        adapter: object,
        auditor_model: object,
        threshold: float = 0.5,
    ) -> None:
        self.primary_model = adapter
        self.auditor_model = auditor_model
        self.threshold = threshold

    def route(self, X: np.ndarray) -> dict:
        """Route samples to AI or suppression.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            A dict with keys:
                - "show_mask" (np.ndarray[bool]): True where AI is shown.
                - "suppress_mask" (np.ndarray[bool]): True where suppressed.
                - "p_wrong" (np.ndarray[float]): P(primary wrong) per sample.
                - "ai_predictions" (np.ndarray): Primary model predictions.
        """
        scores = self.auditor_model.p_wrong(X, self.primary_model)
        suppress_mask = scores >= self.threshold
        show_mask = ~suppress_mask
        ai_predictions = self.primary_model.predict(X)
        return {
            "show_mask": show_mask,
            "suppress_mask": suppress_mask,
            "p_wrong": scores,
            "ai_predictions": ai_predictions,
        }

    def set_threshold(self, tau: float) -> None:
        """Update the suppression threshold.

        Args:
            tau: New threshold value. Must be strictly between 0 and 1.

        Raises:
            ValueError: If tau is not in the open interval (0, 1).
        """
        if not (0.0 < tau < 1.0):
            raise ValueError(
                f"Threshold must be in (0.0, 1.0), got {tau}."
            )
        self.threshold = tau

    def sweep_thresholds(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        human_accuracy: float = 0.72,
        taus: list | None = None,
    ) -> pd.DataFrame:
        """Sweep over suppression thresholds and compute joint accuracy metrics.

        For each threshold, computes:
            - suppression_rate: fraction of samples suppressed.
            - ai_accuracy_on_shown: accuracy of AI on shown (non-suppressed) samples.
            - joint_accuracy: accuracy_on_shown * (1 - rate) + human_accuracy * rate.
            - accuracy_gain: joint_accuracy - ai_only_accuracy.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y_true: True labels of shape (n_samples,).
            human_accuracy: Simulated human accuracy on suppressed cases.
                Defaults to 0.72.
            taus: List of threshold values to sweep. Defaults to
                np.linspace(0.1, 0.9, 17).

        Returns:
            A DataFrame with columns: tau, suppression_rate,
            ai_accuracy_on_shown, joint_accuracy, accuracy_gain.
        """
        if taus is None:
            taus = np.linspace(0.1, 0.9, 17).tolist()

        ai_predictions = self.primary_model.predict(X)
        ai_only_accuracy = float(np.mean(ai_predictions == y_true))
        scores = self.auditor_model.p_wrong(X, self.primary_model)

        rows = []
        for tau in taus:
            suppress = scores >= tau
            show = ~suppress
            rate = float(np.mean(suppress))
            n_shown = int(np.sum(show))
            if n_shown > 0:
                ai_acc_on_shown = float(
                    np.mean(ai_predictions[show] == y_true[show])
                )
            else:
                ai_acc_on_shown = 0.0
            joint = ai_acc_on_shown * (1 - rate) + human_accuracy * rate
            gain = joint - ai_only_accuracy
            rows.append(
                {
                    "tau": tau,
                    "suppression_rate": rate,
                    "ai_accuracy_on_shown": ai_acc_on_shown,
                    "joint_accuracy": joint,
                    "accuracy_gain": gain,
                }
            )
        return pd.DataFrame(rows)

    def best_threshold(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        human_accuracy: float = 0.72,
    ) -> float:
        """Find the threshold that maximises accuracy gain.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y_true: True labels of shape (n_samples,).
            human_accuracy: Simulated human accuracy. Defaults to 0.72.

        Returns:
            The tau value that yields the highest accuracy_gain.
        """
        df = self.sweep_thresholds(X, y_true, human_accuracy=human_accuracy)
        best_row = df.loc[df["accuracy_gain"].idxmax()]
        best_tau = float(best_row["tau"])
        logger.info(
            "Best threshold: tau=%.3f  (accuracy_gain=%.4f)",
            best_tau,
            float(best_row["accuracy_gain"]),
        )
        return best_tau
