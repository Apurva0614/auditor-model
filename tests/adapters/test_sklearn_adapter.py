"""
Tests for SklearnAdapter.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from auditorai.adapters.base import wrap
from auditorai.adapters.sklearn_adapter import SklearnAdapter


@pytest.fixture
def small_dataset():
    """200-sample, 10-feature binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    return X, y


@pytest.fixture
def fitted_rf(small_dataset):
    """Fitted RandomForestClassifier."""
    X, y = small_dataset
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    return rf


def test_wrap_auto_detects_sklearn(fitted_rf):
    """wrap() with auto detection returns SklearnAdapter for sklearn models."""
    adapter = wrap(fitted_rf)
    assert isinstance(adapter, SklearnAdapter)


def test_predict_shape(fitted_rf, small_dataset):
    """predict() returns array of shape (n,)."""
    X, y = small_dataset
    adapter = SklearnAdapter(fitted_rf)
    preds = adapter.predict(X)
    assert preds.shape == (200,)


def test_predict_proba_shape(fitted_rf, small_dataset):
    """predict_proba() returns array of shape (n, n_classes)."""
    X, y = small_dataset
    adapter = SklearnAdapter(fitted_rf)
    probas = adapter.predict_proba(X)
    assert probas.shape == (200, 2)


def test_predict_proba_sums_to_one(fitted_rf, small_dataset):
    """Each row of predict_proba sums to 1.0."""
    X, y = small_dataset
    adapter = SklearnAdapter(fitted_rf)
    probas = adapter.predict_proba(X)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)


def test_calibration_applied_when_no_predict_proba(small_dataset):
    """SVC without probability=True gets wrapped for calibration."""
    X, y = small_dataset
    svc = SVC(probability=False, random_state=42)
    svc.fit(X, y)
    adapter = SklearnAdapter(svc, calibrate=True)
    # SVC without probability has decision_function but no predict_proba
    # The adapter should handle this via the fallback path
    # Fit calibrated model
    adapter.fit(X, y)
    probas = adapter.predict_proba(X)
    assert probas.shape == (200, 2)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-3)


def test_validate_probas_raises_on_bad_input(fitted_rf):
    """validate_probas raises ValueError for invalid probability matrices."""
    adapter = SklearnAdapter(fitted_rf)

    # Not summing to 1
    bad_probas = np.array([[0.5, 0.3], [0.4, 0.4]])
    with pytest.raises(ValueError, match="rows must sum to 1.0"):
        adapter.validate_probas(bad_probas)

    # Not 2D
    bad_1d = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="2D array"):
        adapter.validate_probas(bad_1d)

    # Negative values
    bad_neg = np.array([[1.2, -0.2], [0.5, 0.5]])
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        adapter.validate_probas(bad_neg)


def test_sklearn_pipeline_works(small_dataset):
    """Wrapping a sklearn Pipeline works correctly."""
    X, y = small_dataset
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, max_iter=1000)),
    ])
    pipe.fit(X, y)

    adapter = wrap(pipe)
    assert isinstance(adapter, SklearnAdapter)
    preds = adapter.predict(X)
    assert preds.shape == (200,)
    probas = adapter.predict_proba(X)
    assert probas.shape == (200, 2)
