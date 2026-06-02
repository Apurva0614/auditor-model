"""
AuditorAI adapters package.

Exports: ModelAdapter, SklearnAdapter, PyTorchAdapter,
         HuggingFaceAdapter, APIAdapter, wrap

Uses lazy imports so that missing optional dependencies
(torch, transformers, openai, anthropic) do not cause
ImportError at package import time. Only raises ImportError
when the specific adapter is actually instantiated.
"""

from auditorai.adapters.base import ModelAdapter, wrap

# Lazy imports for optional-dependency adapters
_SKLEARN_ADAPTER = None
_PYTORCH_ADAPTER = None
_HUGGINGFACE_ADAPTER = None
_API_ADAPTER = None


def __getattr__(name):
    """Lazy import mechanism for adapter classes."""
    if name == "SklearnAdapter":
        from auditorai.adapters.sklearn_adapter import SklearnAdapter
        return SklearnAdapter
    if name == "PyTorchAdapter":
        from auditorai.adapters.pytorch_adapter import PyTorchAdapter
        return PyTorchAdapter
    if name == "HuggingFaceAdapter":
        from auditorai.adapters.huggingface_adapter import HuggingFaceAdapter
        return HuggingFaceAdapter
    if name == "APIAdapter":
        from auditorai.adapters.api_adapter import APIAdapter
        return APIAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelAdapter",
    "SklearnAdapter",
    "PyTorchAdapter",
    "HuggingFaceAdapter",
    "APIAdapter",
    "wrap",
]
