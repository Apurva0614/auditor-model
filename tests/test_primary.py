"""
Tests for PrimaryModel.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.primary_model import PrimaryModel


@pytest.fixture
def small_dataset():
    """200-sample, 10-feature binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    return X, y


def test_fit_random_forest(small_dataset):
    X, y = small_dataset
    model = PrimaryModel("random_forest")
    model.fit(X, y)
    assert model.pipeline_ is not None


def test_fit_gradient_boosting(small_dataset):
    X, y = small_dataset
    model = PrimaryModel("gradient_boosting")
    model.fit(X, y)
    assert model.pipeline_ is not None


def test_fit_logistic(small_dataset):
    X, y = small_dataset
    model = PrimaryModel("logistic")
    model.fit(X, y)
    assert model.pipeline_ is not None


def test_invalid_model_type():
    with pytest.raises(ValueError):
        PrimaryModel("invalid_model")


def test_predict_shape(small_dataset):
    X, y = small_dataset
    model = PrimaryModel("random_forest")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (200,)


def test_predict_proba_shape(small_dataset):
    X, y = small_dataset
    model = PrimaryModel("random_forest")
    model.fit(X, y)
    probas = model.predict_proba(X)
    assert probas.shape == (200, 2)


def test_predict_proba_sums_to_one(small_dataset):
    X, y = small_dataset
    model = PrimaryModel("random_forest")
    model.fit(X, y)
    probas = model.predict_proba(X)
    sums = probas.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(200), atol=1e-6)


def test_evaluate_keys(small_dataset):
    X, y = small_dataset
    model = PrimaryModel("random_forest")
    model.fit(X, y)
    result = model.evaluate(X, y)
    assert "accuracy" in result
    assert "roc_auc" in result
    assert "report" in result


def test_evaluate_accuracy_range(small_dataset):
    X, y = small_dataset
    model = PrimaryModel("random_forest")
    model.fit(X, y)
    result = model.evaluate(X, y)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_predict_before_fit_raises():
    model = PrimaryModel("random_forest")
    X = np.random.rand(10, 5)
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_save_load_roundtrip(tmp_path, small_dataset):
    X, y = small_dataset
    model = PrimaryModel("random_forest")
    model.fit(X, y)
    preds_before = model.predict(X)

    path = str(tmp_path / "primary.joblib")
    model.save(path)

    new_model = PrimaryModel("random_forest")
    new_model.load(path)
    preds_after = new_model.predict(X)

    np.testing.assert_array_equal(preds_before, preds_after)
