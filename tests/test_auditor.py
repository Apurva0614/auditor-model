"""
Tests for AuditorModel (updated imports for auditorai package).
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from auditorai.core.auditor import AuditorModel
from auditorai.adapters.sklearn_adapter import SklearnAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def dataset():
    """400-sample, 10-feature dataset with 10% label noise."""
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=5, flip_y=0.1, random_state=42
    )
    return X, y


@pytest.fixture
def trained_primary(dataset):
    """SklearnAdapter wrapping a fitted model, trained on the first 200 samples."""
    X, y = dataset
    base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    calibrated = CalibratedClassifierCV(base_clf, cv=3)
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])
    pipeline.fit(X[:200], y[:200])
    return SklearnAdapter(pipeline)


def test_build_features_shape(dataset, trained_primary):
    X, y = dataset
    auditor = AuditorModel()
    probas = trained_primary.predict_proba(X[:50])
    augmented = auditor._build_features(X[:50], probas)
    # 10 original features + 4 derived = 14
    assert augmented.shape == (50, 14)


def test_build_features_no_mutation(dataset, trained_primary):
    X, y = dataset
    X_copy = X[:50].copy()
    auditor = AuditorModel()
    probas = trained_primary.predict_proba(X[:50])
    _ = auditor._build_features(X[:50], probas)
    np.testing.assert_array_equal(X[:50], X_copy)


def test_p_wrong_range(dataset, trained_primary):
    X, y = dataset
    auditor = AuditorModel()
    # train auditor on held-out second half
    auditor.fit(X[200:], y[200:], trained_primary)
    scores = auditor.p_wrong(X[200:], trained_primary)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_fit_produces_pipeline(dataset, trained_primary):
    X, y = dataset
    auditor = AuditorModel()
    auditor.fit(X[200:], y[200:], trained_primary)
    assert auditor.pipeline_ is not None


def test_suppression_is_bool_array(dataset, trained_primary):
    X, y = dataset
    auditor = AuditorModel()
    auditor.fit(X[200:], y[200:], trained_primary)
    suppress = auditor.predict_suppression(X[200:], trained_primary)
    assert suppress.dtype == bool


def test_higher_threshold_fewer_suppressions(dataset, trained_primary):
    X, y = dataset
    auditor_low = AuditorModel(threshold=0.3)
    auditor_low.fit(X[200:], y[200:], trained_primary)
    low_count = int(np.sum(auditor_low.predict_suppression(X[200:], trained_primary)))

    auditor_high = AuditorModel(threshold=0.7)
    auditor_high.pipeline_ = auditor_low.pipeline_  # same underlying model
    auditor_high.threshold = 0.7
    high_count = int(np.sum(auditor_high.predict_suppression(X[200:], trained_primary)))

    assert high_count <= low_count


def test_evaluate_keys(dataset, trained_primary):
    X, y = dataset
    auditor = AuditorModel()
    auditor.fit(X[200:], y[200:], trained_primary)
    result = auditor.evaluate(X[200:], y[200:], trained_primary)
    for key in ("threshold", "suppression_rate", "auroc", "precision", "recall"):
        assert key in result


def test_auroc_above_chance(dataset, trained_primary):
    X, y = dataset
    auditor = AuditorModel()
    auditor.fit(X[200:], y[200:], trained_primary)
    result = auditor.evaluate(X[200:], y[200:], trained_primary)
    assert result["auroc"] >= 0.4


def test_save_load_roundtrip(tmp_path, dataset, trained_primary):
    X, y = dataset
    auditor = AuditorModel()
    auditor.fit(X[200:], y[200:], trained_primary)
    scores_before = auditor.p_wrong(X[200:], trained_primary)

    path = str(tmp_path / "auditor.joblib")
    auditor.save(path)

    new_auditor = AuditorModel()
    new_auditor.load(path)
    scores_after = new_auditor.p_wrong(X[200:], trained_primary)

    np.testing.assert_allclose(scores_before, scores_after, rtol=1e-5)
