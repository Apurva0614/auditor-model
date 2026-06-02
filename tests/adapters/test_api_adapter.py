"""
Tests for APIAdapter.
All tests use unittest.mock — no real API calls are made.
"""

import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from auditorai.adapters.api_adapter import APIAdapter


def test_openai_provider_detected():
    """'gpt-4o' auto-detects as openai provider."""
    adapter = APIAdapter(model_name="gpt-4o", api_key="fake-key",
                          n_classes=2, parse_response=lambda x: (0, 0.5))
    assert adapter.provider == "openai"


def test_anthropic_provider_detected():
    """'claude-sonnet-4-6' auto-detects as anthropic provider."""
    adapter = APIAdapter(model_name="claude-sonnet-4-6", api_key="fake-key",
                          n_classes=2, parse_response=lambda x: (0, 0.5))
    assert adapter.provider == "anthropic"


def test_api_key_from_env():
    """API key is read from environment if not passed directly."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-env-key"}):
        adapter = APIAdapter(model_name="gpt-4o", n_classes=2,
                              parse_response=lambda x: (0, 0.5))
        assert adapter.api_key == "sk-test-env-key"


def test_missing_api_key_raises():
    """EnvironmentError raised when no API key found."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove both keys from env
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("ANTHROPIC_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="No API key found"):
                APIAdapter(model_name="gpt-4o", n_classes=2,
                           parse_response=lambda x: (0, 0.5))


def test_probas_from_parse_binary():
    """Binary class: class=1, conf=0.9, n=2 -> [0.1, 0.9]."""
    adapter = APIAdapter(model_name="gpt-4o", api_key="fake",
                          n_classes=2, parse_response=lambda x: (1, 0.9))
    probas = adapter._probas_from_parse(1, 0.9)
    np.testing.assert_allclose(probas, [0.1, 0.9], atol=1e-10)
    assert abs(probas.sum() - 1.0) < 1e-10


def test_probas_from_parse_multiclass():
    """Multiclass: class=2, conf=0.8, n=3 -> probas[2]==0.8, rest sums to 0.2."""
    adapter = APIAdapter(model_name="gpt-4o", api_key="fake",
                          n_classes=3, parse_response=lambda x: (2, 0.8))
    probas = adapter._probas_from_parse(2, 0.8)
    assert probas[2] == pytest.approx(0.8)
    assert probas[0] == pytest.approx(0.1)
    assert probas[1] == pytest.approx(0.1)
    assert probas.sum() == pytest.approx(1.0)


def test_predict_calls_parse_response():
    """Mock API call and verify the parser is invoked."""
    parser = MagicMock(return_value=(1, 0.85))

    adapter = APIAdapter(model_name="gpt-4o", api_key="fake-key",
                          n_classes=2, parse_response=parser,
                          batch_size=1)

    # Mock the API call to return a fixed string
    with patch.object(adapter, "_call_api", return_value="Class: 1, Confidence: 0.85"):
        preds = adapter.predict(["test input"])
        assert preds[0] == 1
        parser.assert_called_once_with("Class: 1, Confidence: 0.85")


def test_exponential_backoff_on_rate_limit():
    """Verify retry logic is triggered on rate limit errors."""
    call_count = 0

    def mock_call(text):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception("Rate limit exceeded (429)")
        return "Class: 0, Confidence: 0.7"

    adapter = APIAdapter(model_name="gpt-4o", api_key="fake-key",
                          n_classes=2,
                          parse_response=lambda x: (0, 0.7),
                          batch_size=1)
    adapter.base_delay = 0.01  # Speed up test

    with patch.object(adapter, "_call_openai", side_effect=mock_call):
        preds = adapter.predict(["test"])
        assert preds[0] == 0
        assert call_count == 3  # 2 retries + 1 success
