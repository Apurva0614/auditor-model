"""
Tests for SklearnAdapter (replaces PrimaryModel tests).
Updated imports for auditorai package.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from auditorai.adapters.sklearn_adapter import SklearnAdapter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_adapter(model_type: str = "random_forest"):
    """Build a SklearnAdapter wrapping a calibrated pipeline."""
    if model_type == "random_forest":
        base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "gradient_boosting":
        base_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        base_clf = LogisticRegression(max_iter=1000, random_state=42)

    calibrated = CalibratedClassifierCV(base_clf, cv=3)
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])
    return SklearnAdapter(pipeline)


@pytest.fixture
def small_dataset():
    """200-sample, 10-feature binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    return X, y


def test_fit_random_forest(small_dataset):
    X, y = small_dataset
    adapter = _build_adapter("random_forest")
    adapter.fit(X, y)
    preds = adapter.predict(X)
    assert preds is not None


def test_fit_gradient_boosting(small_dataset):
    X, y = small_dataset
    adapter = _build_adapter("gradient_boosting")
    adapter.fit(X, y)
    preds = adapter.predict(X)
    assert preds is not None


def test_fit_logistic(small_dataset):
    X, y = small_dataset
    adapter = _build_adapter("logistic")
    adapter.fit(X, y)
    preds = adapter.predict(X)
    assert preds is not None


def test_invalid_model_type():
    with pytest.raises(ValueError):
        # Test that invalid model_type for the old PrimaryModel
        # equivalent would fail — we test via direct instantiation
        from src.primary_model import PrimaryModel
        PrimaryModel("invalid_model")


def test_predict_shape(small_dataset):
    X, y = small_dataset
    adapter = _build_adapter("random_forest")
    adapter.fit(X, y)
    preds = adapter.predict(X)
    assert preds.shape == (200,)


def test_predict_proba_shape(small_dataset):
    X, y = small_dataset
    adapter = _build_adapter("random_forest")
    adapter.fit(X, y)
    probas = adapter.predict_proba(X)
    assert probas.shape == (200, 2)


def test_predict_proba_sums_to_one(small_dataset):
    X, y = small_dataset
    adapter = _build_adapter("random_forest")
    adapter.fit(X, y)
    probas = adapter.predict_proba(X)
    sums = probas.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(200), atol=1e-6)


def test_evaluate_keys(small_dataset):
    X, y = small_dataset
    # Use the old PrimaryModel to keep backward compatibility test
    from src.primary_model import PrimaryModel
    model = PrimaryModel("random_forest")
    model.fit(X, y)
    result = model.evaluate(X, y)
    assert "accuracy" in result
    assert "roc_auc" in result
    assert "report" in result


def test_evaluate_accuracy_range(small_dataset):
    X, y = small_dataset
    from src.primary_model import PrimaryModel
    model = PrimaryModel("random_forest")
    model.fit(X, y)
    result = model.evaluate(X, y)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_predict_before_fit_raises():
    from src.primary_model import PrimaryModel
    model = PrimaryModel("random_forest")
    X = np.random.rand(10, 5)
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_save_load_roundtrip(tmp_path, small_dataset):
    X, y = small_dataset
    from src.primary_model import PrimaryModel
    model = PrimaryModel("random_forest")
    model.fit(X, y)
    preds_before = model.predict(X)

    path = str(tmp_path / "primary.joblib")
    model.save(path)

    new_model = PrimaryModel("random_forest")
    new_model.load(path)
    preds_after = new_model.predict(X)

    np.testing.assert_array_equal(preds_before, preds_after)
