"""
Auditor model that predicts when the primary model is likely to be wrong.

CRITICAL DESIGN RULE:
The auditor MUST be trained on a held-out validation set — never the same
data the primary model was trained on. On training data the primary model
has memorised the answers; its errors there are unrepresentative of its
true generalisation behaviour and will produce a degenerate auditor.
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from auditorai.utils.logging import get_logger

logger = get_logger(__name__)


class AuditorModel:
    """Predicts when the primary model is likely wrong on a given input.

    The auditor learns to detect unreliable primary-model predictions by
    combining the original features with derived uncertainty signals
    (confidence, entropy, margin).

    CRITICAL: Train the auditor only on a held-out validation set.
    If trained on the same data as the primary model, the primary's errors
    are unrepresentative because it has memorised the training examples.

    Args:
        threshold: Probability threshold above which a prediction is
            suppressed. Defaults to 0.5.
    """

    def __init__(self, threshold: float = 0.5, feature_fn: object = None) -> None:
        self.threshold = threshold
        self.feature_fn = feature_fn
        self.pipeline_: Pipeline | None = None

    def _build_features(
        self, X: np.ndarray, primary_probas: np.ndarray
    ) -> np.ndarray:
        """Augment X with uncertainty-derived columns and optional custom features.

        CRITICAL: X is never modified in place. A new array is returned.

        The default four appended features are:
            1. confidence:      max probability per row,          shape (n,1)
            2. predicted_class: argmax per row cast to float,     shape (n,1)
            3. entropy:         -sum(p * log(p + 1e-10)) per row, shape (n,1)
            4. margin:          top-1 prob - top-2 prob,          shape (n,1)

        If feature_fn is provided, its output is concatenated to the default 4 features.

        Args:
            X: Original feature array of shape (n_samples, n_features).
            primary_probas: Class probability array of shape
                (n_samples, n_classes).

        Returns:
            Augmented array of shape (n_samples, n_features + 4 + n_custom_features).
        """
        n = X.shape[0]

        confidence = np.max(primary_probas, axis=1, keepdims=True)  # (n, 1)
        predicted_class = np.argmax(primary_probas, axis=1, keepdims=True).astype(
            np.float64
        )  # (n, 1)

        # Shannon entropy: -sum(p * log(p + eps))
        entropy = -np.sum(
            primary_probas * np.log(primary_probas + 1e-10), axis=1, keepdims=True
        )  # (n, 1)

        # Margin between top-1 and top-2 probability
        if primary_probas.shape[1] >= 2:
            sorted_probas = np.sort(primary_probas, axis=1)[:, ::-1]
            margin = (sorted_probas[:, 0] - sorted_probas[:, 1]).reshape(n, 1)
        else:
            margin = confidence  # degenerate single-class case

        base_features = [X, confidence, predicted_class, entropy, margin]

        if self.feature_fn is not None:
            predictions = np.argmax(primary_probas, axis=1)
            custom_feats = self.feature_fn(primary_probas, predictions)
            custom_feats = np.asarray(custom_feats)
            if custom_feats.ndim == 1:
                custom_feats = custom_feats.reshape(-1, 1)
            base_features.append(custom_feats)

        return np.hstack(base_features)

    def fit(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        primary_model: object,
    ) -> None:
        """Fit the auditor on a held-out validation set.

        CRITICAL: X must be a held-out validation set — never the training
        data used to fit the primary model. The primary model memorises its
        training examples, so its errors there are unrepresentative of
        real-world failure modes.

        Steps:
            1. Obtain primary model predictions and probabilities on X.
            2. Derive auditor labels: 1 where primary is wrong, 0 where correct.
            3. Augment X with uncertainty features.
            4. Fit a GradientBoostingClassifier pipeline on the augmented data.

        Args:
            X: Held-out validation features (n_samples, n_features).
            y_true: True labels for the validation samples (n_samples,).
            primary_model: A fitted model with predict() and predict_proba()
                methods (ModelAdapter or any compatible object).
        """
        primary_preds = primary_model.predict(X)
        primary_probas = primary_model.predict_proba(X)

        auditor_labels = (primary_preds != y_true).astype(int)
        error_rate = float(np.mean(auditor_labels))
        logger.info(
            "Auditor training — primary error rate on validation set: %.3f",
            error_rate,
        )

        augmented_X = self._build_features(X, primary_probas)

        self.pipeline_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=42,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(augmented_X, auditor_labels)
        logger.info("AuditorModel fitted on %d validation samples.", len(X))

    @classmethod
    def from_errors(cls, X: np.ndarray,
                    error_mask: np.ndarray,
                    primary_probas: np.ndarray,
                    threshold: float = 0.5) -> "AuditorModel":
        """
        Alternative constructor for when you already know which
        predictions were errors (e.g. from a production logging system).

        Args:
          X: Feature matrix used by the primary model.
          error_mask: Boolean array, True = primary model was wrong.
          primary_probas: Probability outputs from the primary model.
          threshold: Suppression threshold.

        Returns a fitted AuditorModel without needing a ModelAdapter.
        Useful for offline training from logged predictions.
        """
        auditor = cls(threshold=threshold)
        auditor_labels = error_mask.astype(int)

        augmented_X = auditor._build_features(X, primary_probas)

        auditor.pipeline_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=42,
                    ),
                ),
            ]
        )
        auditor.pipeline_.fit(augmented_X, auditor_labels)
        logger.info("AuditorModel fitted from error mask on %d samples.", len(X))
        return auditor

    def p_wrong(
        self, X: np.ndarray, primary_model: object
    ) -> np.ndarray:
        """Estimate P(primary model is wrong) for each sample.

        Args:
            X: Feature array of shape (n_samples, n_features).
            primary_model: A fitted model with predict_proba()
                (ModelAdapter or any compatible object).

        Returns:
            Float array of shape (n_samples,) with values in [0, 1].
        """
        primary_probas = primary_model.predict_proba(X)
        augmented_X = self._build_features(X, primary_probas)
        return self.pipeline_.predict_proba(augmented_X)[:, 1]

    def predict_suppression(
        self, X: np.ndarray, primary_model: object
    ) -> np.ndarray:
        """Determine which samples should be suppressed (withheld from the AI).

        A sample is suppressed when P(wrong) >= self.threshold.

        Args:
            X: Feature array of shape (n_samples, n_features).
            primary_model: A fitted model with predict_proba().

        Returns:
            Boolean array of shape (n_samples,). True means suppress.
        """
        return self.p_wrong(X, primary_model) >= self.threshold

    def evaluate(
        self, X: np.ndarray, y_true: np.ndarray, primary_model: object
    ) -> dict:
        """Evaluate auditor performance.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y_true: True labels of shape (n_samples,).
            primary_model: A fitted model with predict() and predict_proba().

        Returns:
            A dict with keys:
                - "threshold" (float): Current suppression threshold.
                - "suppression_rate" (float): Fraction of samples suppressed.
                - "auroc" (float): ROC-AUC for predicting primary errors.
                - "precision" (float): Of suppressed samples, fraction that
                    were real errors. 0.0 if nothing is suppressed.
                - "recall" (float): Of real errors, fraction suppressed.
                    0.0 if there are no errors.
        """
        scores = self.p_wrong(X, primary_model)
        suppress = scores >= self.threshold
        primary_preds = primary_model.predict(X)
        real_errors = (primary_preds != y_true).astype(int)

        suppression_rate = float(np.mean(suppress))
        auroc = float(roc_auc_score(real_errors, scores)) if len(np.unique(real_errors)) > 1 else 0.5

        suppress_int = suppress.astype(int)
        if suppress_int.sum() == 0:
            precision = 0.0
        else:
            precision = float(precision_score(real_errors, suppress_int, zero_division=0))

        if real_errors.sum() == 0:
            recall = 0.0
        else:
            recall = float(recall_score(real_errors, suppress_int, zero_division=0))

        return {
            "threshold": self.threshold,
            "suppression_rate": suppression_rate,
            "auroc": auroc,
            "precision": precision,
            "recall": recall,
        }

    def save(self, path: str) -> None:
        """Serialize the auditor to disk.

        Creates parent directories if they do not exist.

        Args:
            path: Destination file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        logger.info("AuditorModel saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved AuditorModel from disk.

        Args:
            path: Path to the serialized model file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Auditor model file not found at: {path}")
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        logger.info("AuditorModel loaded from %s", path)
