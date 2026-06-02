"""
Tests for PyTorchAdapter.
Skip all tests if torch is not installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from auditorai.adapters.base import wrap
from auditorai.adapters.pytorch_adapter import PyTorchAdapter


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP for testing."""
    def __init__(self, input_dim=10, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def simple_model():
    """Return a simple MLP model."""
    model = SimpleMLP(input_dim=10, n_classes=2)
    model.eval()
    return model


@pytest.fixture
def sample_data():
    """Return sample numpy data."""
    np.random.seed(42)
    X = np.random.randn(50, 10).astype(np.float32)
    return X


def test_wrap_auto_detects_pytorch(simple_model):
    """wrap() with auto detection returns PyTorchAdapter for nn.Module."""
    adapter = wrap(simple_model, n_classes=2)
    assert isinstance(adapter, PyTorchAdapter)


def test_predict_shape(simple_model, sample_data):
    """predict() returns array of shape (n,)."""
    adapter = PyTorchAdapter(simple_model, n_classes=2, device="cpu")
    preds = adapter.predict(sample_data)
    assert preds.shape == (50,)
    assert preds.dtype in (np.int32, np.int64)


def test_predict_proba_sums_to_one(simple_model, sample_data):
    """predict_proba rows sum to 1.0."""
    adapter = PyTorchAdapter(simple_model, n_classes=2, device="cpu")
    probas = adapter.predict_proba(sample_data)
    assert probas.shape == (50, 2)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


def test_device_auto_detection(simple_model):
    """device='auto' resolves to a valid string."""
    adapter = PyTorchAdapter(simple_model, n_classes=2, device="auto")
    assert adapter.device in ("cpu", "cuda", "mps")


def test_logits_converted_to_proba(simple_model, sample_data):
    """Raw logits produce valid probability matrix via softmax."""
    adapter = PyTorchAdapter(simple_model, n_classes=2, device="cpu",
                              output_type="logits")
    probas = adapter.predict_proba(sample_data)
    assert (probas >= 0).all()
    assert (probas <= 1).all()
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


def test_batched_inference(simple_model):
    """Large input (1000 rows) runs without error using batching."""
    np.random.seed(42)
    X_large = np.random.randn(1000, 10).astype(np.float32)
    adapter = PyTorchAdapter(simple_model, n_classes=2, device="cpu",
                              batch_size=128)
    probas = adapter.predict_proba(X_large)
    assert probas.shape == (1000, 2)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)
