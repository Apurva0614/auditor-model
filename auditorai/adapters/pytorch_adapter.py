"""
PyTorch adapter for AuditorAI.

Wraps any torch.nn.Module for use with the auditor system.
Handles device management, batched inference, and logit-to-probability
conversion automatically.
"""

import numpy as np

from auditorai.adapters.base import ModelAdapter


class PyTorchAdapter(ModelAdapter):
    """
    Wraps any PyTorch nn.Module for use with AuditorAI.

    Handles:
      - Raw logit output -> softmax conversion
      - GPU/CPU device management automatically
      - Batched inference to avoid OOM on large datasets
      - Both classification heads and full models

    Usage:
      import torch.nn as nn
      from auditorai import AuditorSystem, wrap

      model = MyClassifier()
      model.load_state_dict(torch.load("model.pt"))
      adapter = wrap(model, adapter_type="pytorch",
                     n_classes=3, device="cuda")
      system = AuditorSystem(adapter)

    Args:
      model: A fitted torch.nn.Module in eval mode.
      n_classes: Number of output classes.
      device: "cpu", "cuda", or "auto" (detects GPU automatically).
      batch_size: Inference batch size. Default 256.
      output_type: "logits" (apply softmax) or "proba" (already normalized).
      input_transform: Optional callable applied to X before forward pass.
                       Use this to convert numpy arrays to tensors with
                       custom preprocessing.
    """

    def __init__(self, model, n_classes: int,
                 device: str = "auto",
                 batch_size: int = 256,
                 output_type: str = "logits",
                 input_transform=None):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyTorchAdapter requires torch. Install it with: pip install torch"
            )

        self.model = model
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.output_type = output_type
        self.input_transform = input_transform
        self.device = self._get_device(device)

        import torch as _torch
        self._torch = _torch

        # Move model to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def _get_device(self, device: str) -> str:
        """Detects available device. 'auto' prefers CUDA then MPS then CPU."""
        import torch as _torch

        if device == "auto":
            if _torch.cuda.is_available():
                return "cuda"
            elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _to_tensor(self, X):
        """
        Converts X to a float tensor on the right device.
        Handles: np.ndarray, pd.DataFrame, list, existing tensors.
        If input_transform is set, applies it first.
        """
        torch = self._torch

        if self.input_transform is not None:
            X = self.input_transform(X)

        if isinstance(X, torch.Tensor):
            return X.float().to(self.device)

        # Handle pandas DataFrame
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                X = X.values
        except ImportError:
            pass

        if isinstance(X, np.ndarray):
            return torch.from_numpy(X.astype(np.float32)).to(self.device)

        if isinstance(X, list):
            return torch.tensor(X, dtype=torch.float32).to(self.device)

        raise TypeError(
            f"Cannot convert {type(X).__name__} to tensor. "
            f"Pass np.ndarray, list, pd.DataFrame, torch.Tensor, "
            f"or set input_transform."
        )

    def _batched_forward(self, X) -> np.ndarray:
        """
        Runs model.forward() in batches. Returns raw outputs as numpy.
        Sets model.eval() and uses torch.no_grad() automatically.
        """
        torch = self._torch
        tensor_X = self._to_tensor(X)

        self.model.eval()
        all_outputs = []

        with torch.no_grad():
            n = tensor_X.shape[0]
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch = tensor_X[start:end]
                output = self.model(batch)
                all_outputs.append(output.cpu().numpy())

        return np.concatenate(all_outputs, axis=0)

    def predict(self, X) -> np.ndarray:
        """Returns argmax of probabilities."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        """
        Applies softmax to logits if output_type=="logits".
        Validates output is a valid probability matrix.
        """
        torch = self._torch
        raw = self._batched_forward(X)

        if self.output_type == "logits":
            # Apply softmax
            tensor_raw = torch.from_numpy(raw)
            probas = torch.nn.functional.softmax(tensor_raw, dim=1).numpy()
        else:
            probas = raw

        self.validate_probas(probas)
        return probas
