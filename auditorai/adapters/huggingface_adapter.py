"""
HuggingFace Transformers adapter for AuditorAI.

Wraps HuggingFace transformers models (both Pipeline and Model+Tokenizer)
for use with the auditor system.
"""

import numpy as np

from auditorai.adapters.base import ModelAdapter


class HuggingFaceAdapter(ModelAdapter):
    """
    Wraps HuggingFace transformers models for use with AuditorAI.

    Supports two modes:
      1. Pipeline mode: pass a transformers.pipeline object directly.
         Handles text-classification, zero-shot-classification.
      2. Model+Tokenizer mode: pass model and tokenizer separately
         for full control over preprocessing.

    Usage (pipeline mode — easiest):
      from transformers import pipeline
      from auditorai import wrap

      pipe = pipeline("text-classification",
          model="distilbert-base-uncased-finetuned-sst-2-english")
      adapter = wrap(pipe, adapter_type="huggingface")
      system = AuditorSystem(adapter)
      system.train(texts_val, y_val)

    Usage (model+tokenizer mode):
      from transformers import AutoModelForSequenceClassification, AutoTokenizer
      model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
      tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
      adapter = HuggingFaceAdapter(model, tokenizer=tokenizer,
                                    max_length=128, device="cuda")

    Args:
      model: A transformers pipeline OR PreTrainedModel.
      tokenizer: Required if model is a PreTrainedModel.
      labels: List of class label strings. If None, inferred from model config.
      max_length: Max token length for tokenizer. Default 512.
      batch_size: Inference batch size. Default 32.
      device: "cpu", "cuda", or "auto".

    X passed to predict/predict_proba must be a list of strings.
    """

    def __init__(self, model, tokenizer=None, labels=None,
                 max_length: int = 512, batch_size: int = 32,
                 device: str = "auto"):
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "HuggingFaceAdapter requires transformers. "
                "Install it with: pip install transformers torch"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length
        self.batch_size = batch_size

        # Detect device
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Infer labels from model config if not provided
        if self.labels is None and not self._is_pipeline():
            if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
                self.labels = [
                    self.model.config.id2label[i]
                    for i in sorted(self.model.config.id2label.keys())
                ]

    def _is_pipeline(self) -> bool:
        """Returns True if self.model is a transformers Pipeline."""
        try:
            import transformers
            return isinstance(self.model, transformers.Pipeline)
        except ImportError:
            return False

    def _pipeline_predict_proba(self, X: list) -> np.ndarray:
        """
        Runs pipeline inference. Converts {"label": ..., "score": ...}
        output format to a probability matrix.
        Handles both single-label and multi-label pipeline outputs.
        """
        results = self.model(X, batch_size=self.batch_size, truncation=True,
                             max_length=self.max_length)

        # Get all labels from the pipeline
        if self.labels is None:
            # Run one sample with top_k=None / return_all_scores to discover labels
            try:
                sample_result = self.model(X[:1], top_k=None, truncation=True,
                                           max_length=self.max_length)
            except TypeError:
                sample_result = self.model(X[:1], truncation=True,
                                           max_length=self.max_length)

            if isinstance(sample_result[0], list):
                self.labels = [item["label"] for item in sample_result[0]]
            else:
                self.labels = [sample_result[0]["label"]]

        # Get full scores for all labels
        try:
            full_results = self.model(X, top_k=None, batch_size=self.batch_size,
                                      truncation=True, max_length=self.max_length)
        except TypeError:
            full_results = results

        n_classes = len(self.labels)
        label_to_idx = {label: i for i, label in enumerate(self.labels)}
        probas = np.zeros((len(X), n_classes))

        for i, result in enumerate(full_results):
            if isinstance(result, list):
                # Multi-label or top_k output: list of dicts
                for item in result:
                    label = item["label"]
                    if label in label_to_idx:
                        probas[i, label_to_idx[label]] = item["score"]
            elif isinstance(result, dict):
                # Single-label output
                label = result["label"]
                score = result["score"]
                if label in label_to_idx:
                    probas[i, label_to_idx[label]] = score
                    # Distribute remaining probability
                    remaining = 1.0 - score
                    other_indices = [j for j in range(n_classes)
                                     if j != label_to_idx[label]]
                    if other_indices:
                        for j in other_indices:
                            probas[i, j] = remaining / len(other_indices)

        # Normalize rows to sum to 1
        row_sums = probas.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        probas = probas / row_sums

        return probas

    def _model_predict_proba(self, X: list) -> np.ndarray:
        """
        Tokenizes X and runs model forward pass in batches.
        Applies softmax to logits. Returns probability matrix.
        """
        import torch

        if self.tokenizer is None:
            raise ValueError(
                "tokenizer is required when using a PreTrainedModel directly. "
                "Pass tokenizer=... to HuggingFaceAdapter."
            )

        self.model.to(self.device)
        self.model.eval()

        all_probas = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                batch_texts = X[start:end]

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits
                probas = torch.nn.functional.softmax(logits, dim=1)
                all_probas.append(probas.cpu().numpy())

        return np.concatenate(all_probas, axis=0)

    def predict(self, X: list) -> np.ndarray:
        """Returns predicted class indices."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: list) -> np.ndarray:
        """Returns class probability matrix."""
        if isinstance(X, str):
            X = [X]

        if self._is_pipeline():
            probas = self._pipeline_predict_proba(X)
        else:
            probas = self._model_predict_proba(X)

        self.validate_probas(probas)
        return probas
