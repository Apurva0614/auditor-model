"""
Abstract base class for model adapters and the universal wrap() function.

Every model — sklearn, PyTorch, HuggingFace, API-based, or custom —
gets wrapped in an adapter that exposes a single unified interface.
The auditor only ever talks to adapters, never to raw models directly.
"""

from abc import ABC, abstractmethod

import numpy as np


class ModelAdapter(ABC):
    """
    Abstract base class for all model adapters.

    Any model — sklearn, PyTorch, HuggingFace, API-based, or custom —
    can be used with AuditorAI by wrapping it in an adapter that
    implements this interface.

    The adapter must implement exactly two methods:
      predict(X)       -> class labels, shape (n,)
      predict_proba(X) -> probabilities, shape (n, n_classes)

    X can be np.ndarray, pd.DataFrame, list of strings, or any format
    your model expects. The adapter is responsible for any preprocessing.

    For models that do not natively output calibrated probabilities
    (e.g. raw logits from PyTorch, or API models that return text),
    the adapter must convert them to valid probability distributions
    (rows sum to 1.0, values in [0, 1]).
    """

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return predicted class labels, shape (n,)."""
        ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """
        Return class probabilities, shape (n, n_classes).
        Each row must sum to 1.0.
        """
        ...

    def validate_probas(self, probas: np.ndarray) -> None:
        """
        Validates that probas is a valid probability matrix.
        Raises ValueError with a clear message if not.
        Checks: shape is 2D, values in [0,1], rows sum to ~1.0.
        """
        if probas.ndim != 2:
            raise ValueError(
                f"predict_proba must return a 2D array, got shape {probas.shape}"
            )
        if not np.allclose(probas.sum(axis=1), 1.0, atol=1e-3):
            bad_rows = np.where(~np.isclose(probas.sum(axis=1), 1.0, atol=1e-3))[0]
            raise ValueError(
                f"predict_proba rows must sum to 1.0. "
                f"Apply softmax or normalize your outputs. "
                f"Rows with bad sums: {bad_rows[:5].tolist()}"
            )
        if (probas < 0).any() or (probas > 1).any():
            raise ValueError(
                "predict_proba values must be in [0, 1]."
            )


def wrap(model, adapter_type: str = "auto", **kwargs) -> ModelAdapter:
    """
    Convenience function. Automatically detects model type and wraps it.

    Usage:
      adapter = wrap(my_sklearn_model)
      adapter = wrap(my_torch_model, adapter_type="pytorch", n_classes=3)
      adapter = wrap("gpt-4o", adapter_type="openai", api_key="sk-...")

    adapter_type options: "auto", "sklearn", "pytorch", "huggingface",
                          "openai", "anthropic", "custom"

    "auto" detection logic:
      1. If model has .predict and .predict_proba attributes -> "sklearn"
      2. If model is a torch.nn.Module -> "pytorch"
      3. If model is a transformers.PreTrainedModel or Pipeline -> "huggingface"
      4. If model is a string starting with "gpt" or "o1" or "o3" -> "openai"
      5. If model is a string starting with "claude" -> "anthropic"
      6. Otherwise raise ValueError with helpful message.

    Args:
        model: The model to wrap. Can be an sklearn estimator, torch module,
               HuggingFace model/pipeline, model name string, or any object.
        adapter_type: Detection mode. "auto" tries to infer the type.
        **kwargs: Additional arguments passed to the specific adapter constructor.

    Returns:
        A ModelAdapter wrapping the given model.

    Raises:
        ValueError: If auto-detection fails and no adapter_type is specified.
    """
    if adapter_type == "auto":
        adapter_type = _detect_type(model)

    if adapter_type == "sklearn":
        from auditorai.adapters.sklearn_adapter import SklearnAdapter
        return SklearnAdapter(model, **kwargs)
    elif adapter_type == "pytorch":
        from auditorai.adapters.pytorch_adapter import PyTorchAdapter
        return PyTorchAdapter(model, **kwargs)
    elif adapter_type == "huggingface":
        from auditorai.adapters.huggingface_adapter import HuggingFaceAdapter
        return HuggingFaceAdapter(model, **kwargs)
    elif adapter_type in ("openai", "anthropic", "custom"):
        from auditorai.adapters.api_adapter import APIAdapter
        if isinstance(model, str):
            return APIAdapter(model_name=model, provider=adapter_type, **kwargs)
        else:
            return APIAdapter(model_name=None, provider=adapter_type, **kwargs)
    else:
        raise ValueError(
            f"Unknown adapter_type '{adapter_type}'. "
            f"Supported: 'auto', 'sklearn', 'pytorch', 'huggingface', "
            f"'openai', 'anthropic', 'custom'."
        )


def _detect_type(model) -> str:
    """
    Auto-detect model type from the model object.

    Returns one of: "sklearn", "pytorch", "huggingface", "openai", "anthropic".
    Raises ValueError if detection fails.
    """
    # Check for string-based API models
    if isinstance(model, str):
        lower = model.lower()
        if lower.startswith("gpt") or lower.startswith("o1") or lower.startswith("o3"):
            return "openai"
        if lower.startswith("claude"):
            return "anthropic"
        raise ValueError(
            f"Cannot auto-detect adapter for string '{model}'. "
            f"Pass adapter_type='openai', 'anthropic', or 'custom' explicitly."
        )

    # Check for torch.nn.Module
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            return "pytorch"
    except ImportError:
        pass

    # Check for HuggingFace transformers Pipeline or PreTrainedModel
    try:
        import transformers
        if isinstance(model, transformers.Pipeline):
            return "huggingface"
        if isinstance(model, transformers.PreTrainedModel):
            return "huggingface"
    except ImportError:
        pass

    # Check for sklearn-compatible (has predict and predict_proba)
    if hasattr(model, "predict") and hasattr(model, "predict_proba"):
        return "sklearn"

    # Check for sklearn-compatible without predict_proba (e.g. SVM)
    if hasattr(model, "predict") and hasattr(model, "fit"):
        return "sklearn"

    raise ValueError(
        f"Cannot auto-detect adapter for {type(model).__name__}. "
        f"Options:\n"
        f"  1. Pass adapter_type='sklearn', 'pytorch', 'huggingface', "
        f"'openai', 'anthropic', or 'custom'\n"
        f"  2. Write a custom adapter:\n"
        f"     class MyAdapter(ModelAdapter):\n"
        f"         def predict(self, X): ...\n"
        f"         def predict_proba(self, X): ...\n"
    )
