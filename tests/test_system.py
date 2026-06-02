"""
Tests for AuditorSystem (updated imports for auditorai package).
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from auditorai.core.system import AuditorSystem, audit
from auditorai.adapters.sklearn_adapter import SklearnAdapter
from auditorai.utils.data import split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def synthetic_data():
    """600-sample, 12-feature dataset with 8% label noise."""
    X, y = make_classification(
        n_samples=600, n_features=12, n_informative=6, flip_y=0.08, random_state=42
    )
    return X, y


def _build_and_fit_adapter(X_train, y_train):
    """Build a fitted SklearnAdapter."""
    base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    calibrated = CalibratedClassifierCV(base_clf, cv=3)
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])
    pipeline.fit(X_train, y_train)
    return SklearnAdapter(pipeline)


def _trained_system(synthetic_data):
    """Helper: returns a trained AuditorSystem and splits."""
    X, y = synthetic_data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    adapter = _build_and_fit_adapter(X_train, y_train)
    system = AuditorSystem(adapter)
    system.train(X_val, y_val)
    return system, X_train, X_val, X_test, y_train, y_val, y_test


def test_audit_returns_system_and_works(synthetic_data):
    X, y = synthetic_data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    adapter = _build_and_fit_adapter(X_train, y_train)
    system = audit(adapter, X_val, y_val, X_test, y_test)
    assert isinstance(system, AuditorSystem)
    assert system.router_ is not None


def test_train_produces_router(synthetic_data):
    system, *_ = _trained_system(synthetic_data)
    assert system.router_ is not None


def test_predict_returns_required_keys(synthetic_data):
    system, _, _, X_test, _, _, _ = _trained_system(synthetic_data)
    result = system.predict(X_test)
    for key in ("show_mask", "suppress_mask", "p_wrong", "ai_predictions"):
        assert key in result


def test_evaluate_returns_required_keys(synthetic_data):
    system, _, _, X_test, _, _, y_test = _trained_system(synthetic_data)
    metrics = system.evaluate(X_test, y_test)
    expected_keys = {
        "ai_only_accuracy",
        "joint_accuracy",
        "accuracy_gain",
        "suppression_rate",
        "n_shown",
        "n_suppressed",
        "auditor_auroc",
        "auditor_precision",
        "auditor_recall",
    }
    assert expected_keys.issubset(metrics.keys())


def test_accuracy_values_in_range(synthetic_data):
    system, _, _, X_test, _, _, y_test = _trained_system(synthetic_data)
    metrics = system.evaluate(X_test, y_test)
    assert 0.0 <= metrics["ai_only_accuracy"] <= 1.0
    assert 0.0 <= metrics["joint_accuracy"] <= 1.0


def test_n_shown_plus_suppressed_equals_total(synthetic_data):
    system, _, _, X_test, _, _, y_test = _trained_system(synthetic_data)
    metrics = system.evaluate(X_test, y_test)
    assert metrics["n_shown"] + metrics["n_suppressed"] == len(X_test)


def test_auto_tune_changes_threshold(synthetic_data):
    system, _, X_val, _, _, y_val, _ = _trained_system(synthetic_data)
    best_tau = system.auto_tune(X_val, y_val)
    # Verify it was applied consistently.
    assert system.router_.threshold == best_tau
    assert system.auditor_.threshold == best_tau


def test_save_load_roundtrip(tmp_path, synthetic_data):
    system, _, _, X_test, _, _, _ = _trained_system(synthetic_data)
    preds_before = system.predict(X_test)["ai_predictions"]

    save_dir = str(tmp_path / "models")
    system.save(save_dir)

    # Need adapter to reload
    new_system = AuditorSystem(system.adapter)
    new_system.load(save_dir)
    preds_after = new_system.predict(X_test)["ai_predictions"]

    np.testing.assert_array_equal(preds_before, preds_after)


def test_predict_before_train_raises():
    from sklearn.ensemble import RandomForestClassifier as RF
    rf = RF(n_estimators=10, random_state=42)
    rf.fit(np.random.rand(20, 5), np.array([0]*10 + [1]*10))
    adapter = SklearnAdapter(rf)
    system = AuditorSystem(adapter)
    X = np.random.rand(10, 5)
    with pytest.raises(RuntimeError):
        system.predict(X)
