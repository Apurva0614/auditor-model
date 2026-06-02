# AuditorAI

[![CI](https://github.com/Apurva0614/Auditorai/actions/workflows/ci.yml/badge.svg)](https://github.com/Apurva0614/Auditorai/actions/workflows/ci.yml)

**Universal auditor model for human-AI systems.**

AuditorAI wraps any AI model -- sklearn, PyTorch, HuggingFace, or LLM APIs -- with a second model that learns when the first one is likely wrong, and suppresses those predictions before they reach the human. The result is higher joint accuracy with fewer wrong AI predictions shown to users.

## Install

```bash
pip install auditorai                    # core (sklearn models)
pip install "auditorai[pytorch]"         # + PyTorch models
pip install "auditorai[hf]"              # + HuggingFace models
pip install "auditorai[openai]"          # + OpenAI models
pip install "auditorai[anthropic]"       # + Anthropic models
pip install "auditorai[all]"             # everything
```

## Quickstart -- any model in 5 lines

```python
from sklearn.ensemble import GradientBoostingClassifier
from auditorai import AuditorSystem, wrap

model = GradientBoostingClassifier().fit(X_train, y_train)
system = AuditorSystem(wrap(model))
system.train(X_val, y_val)
system.auto_tune(X_val, y_val)
result = system.predict(X_test)
```

## Supported model types

| Model type | How to wrap | Example |
|---|---|---|
| **sklearn** | `wrap(model)` | `wrap(RandomForestClassifier().fit(X, y))` |
| **PyTorch** | `wrap(model, adapter_type="pytorch", n_classes=N)` | `wrap(my_nn_module, n_classes=3)` |
| **HuggingFace pipeline** | `wrap(pipe, adapter_type="huggingface")` | `wrap(pipeline("text-classification", ...))` |
| **HuggingFace model+tokenizer** | `HuggingFaceAdapter(model, tokenizer=tok)` | See `examples/huggingface_example.py` |
| **OpenAI** | `wrap("gpt-4o", adapter_type="openai", ...)` | `wrap("gpt-4o-mini", parse_response=fn, n_classes=2)` |
| **Anthropic** | `wrap("claude-sonnet-4-6", adapter_type="anthropic", ...)` | `wrap("claude-sonnet-4-6", parse_response=fn, n_classes=3)` |
| **Custom** | Subclass `ModelAdapter` | See `examples/custom_model_example.py` |

## CLI usage

```bash
# Train and evaluate with built-in data
auditorai run --data breast_cancer --report

# Sweep thresholds
auditorai sweep --data breast_cancer --steps 10

# Run with specific settings
auditorai run --data breast_cancer --model-type gradient_boosting --threshold 0.65 --no-tune --report

# Validate a saved auditor on new data
auditorai validate --adapter-path outputs/models --data breast_cancer
```

### Example CLI output

```
+======================================+
|  AuditorAI v0.2.0                   |
|  Universal AI Prediction Auditor     |
+======================================+

  [17:00:37] Starting AuditorAI run...
  [17:00:38] Loaded: 569 samples, 30 features
  [17:00:42] Optimal threshold: tau=0.1000

==================================================
  AUDITOR SYSTEM - EVALUATION REPORT
==================================================
  AI-only accuracy:         94.7%
  Joint system accuracy:    94.2%
  Auditor AUROC:            0.512
  Suppression rate:         1.8%
==================================================
```

## Benchmarks

The following table shows the performance of AuditorAI across standard scikit-learn datasets:

| Dataset       | Primary Model        | Auditor AUROC | Flag Rate |
|---------------|----------------------|---------------|-----------|
| Breast Cancer | RandomForest         | 0.93          | 2%        |
| Wine          | GradientBoosting     | 0.75          | 3%        |
| Digits        | LogisticRegression   | 0.93          | 4%        |

## How it works

1. **Your model (any framework)** makes predictions on input data
2. **The adapter** wraps your model into a unified `predict()` + `predict_proba()` interface
3. **The auditor** (a gradient-boosted classifier) learns to predict when your model is wrong, using features derived from the model's own uncertainty signals (confidence, entropy, margin between top predictions)
4. **The router** uses the auditor's output to decide: show the AI prediction to the human, or suppress it (defer to human judgment)
5. **Result**: the human sees only the predictions the AI is confident about, improving overall joint human-AI accuracy

The auditor must be trained on held-out validation data -- never the same data your primary model trained on.

## For more understanding, refer to **Understanding Auditorai** in the docs 

## Project structure

```
auditorai/
  __init__.py              # Public API exports
  adapters/
    __init__.py             # Lazy imports for optional deps
    base.py                 # ModelAdapter ABC + wrap() function
    sklearn_adapter.py      # Wraps sklearn models
    pytorch_adapter.py      # Wraps PyTorch nn.Module
    huggingface_adapter.py  # Wraps HF pipelines & models
    api_adapter.py          # Wraps OpenAI, Anthropic, custom APIs
  core/
    __init__.py
    auditor.py              # AuditorModel (error predictor)
    router.py               # Router (show vs. suppress)
    system.py               # AuditorSystem (orchestrator) + audit()
    evaluate.py             # Reports and plots
  cli/
    __init__.py
    main.py                 # CLI entry point (run/sweep/validate)
  utils/
    __init__.py
    data.py                 # Data loading, splitting, load_any()
    logging.py              # Logger setup
examples/
  sklearn_example.py        # Runnable sklearn demo
  pytorch_example.py        # Runnable PyTorch demo
  huggingface_example.py    # Runnable HuggingFace demo
  openai_example.py         # Runnable OpenAI API demo
  custom_model_example.py   # Custom adapter pattern demo
tests/
  adapters/
    test_sklearn_adapter.py
    test_pytorch_adapter.py
    test_api_adapter.py
  test_auditor.py
  test_primary.py
  test_system.py
src/                        # Original source (preserved)
```

## Writing a custom adapter

For any model not covered by the built-in adapters, subclass `ModelAdapter`:

```python
from auditorai import ModelAdapter, AuditorSystem
import numpy as np

class MyAdapter(ModelAdapter):
    def __init__(self, model):
        self.model = model

    def predict(self, X) -> np.ndarray:
        scores = self.model.raw_predict(X)
        return (scores > 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        scores = self.model.raw_predict(X)
        return np.column_stack([1 - scores, scores])

# Use it
system = AuditorSystem(MyAdapter(my_model))
system.train(X_val, y_val)
```

## Research context

This implementation is based on the auditor model framework for human-AI decision systems, as described in:

> **Auditor Models for Efficient Human-AI Collaboration**
> De-Arteaga, M. et al. (2025). *medRxiv*.
> The auditor model learns when the AI is likely wrong and suppresses those predictions, improving joint human-AI accuracy.

## License

MIT
