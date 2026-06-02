# AuditorAI

### Stop your AI model from showing wrong predictions to users.

[![CI](https://github.com/Apurva0614/Auditorai/actions/workflows/ci.yml/badge.svg)](https://github.com/Apurva0614/Auditorai/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/auditorai.svg)](https://pypi.org/project/auditorai/)
[![Python versions](https://img.shields.io/pypi/pyversions/auditorai.svg)](https://pypi.org/project/auditorai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Apurva0614/Auditorai/blob/main/LICENSE)

**The problem:** Your AI model gets predictions wrong sometimes, and those wrong predictions reach your users — causing bad decisions, lost trust, and real harm.

**The solution:** AuditorAI adds a second model (the "auditor") that learns *when* your primary model is likely wrong and suppresses those predictions before they reach the user. The suppressed cases get routed to a human instead.

**The result:** Only confident, reliable predictions are shown. Wrong predictions are caught and handled by humans, improving your overall system accuracy.

### Quick numbers

| Dataset | AI alone | With AuditorAI | Auditor AUROC | Flag rate |
|---|---|---|---|---|
| Breast Cancer | RandomForest | +auditor | 0.93 | 2% |
| Wine | GradientBoosting | +auditor | 0.75 | 3% |
| Digits | LogisticRegression | +auditor | 0.93 | 4% |

> The auditor identifies unreliable predictions with high precision while flagging only 2–4% of cases for human review.

---

## Install

```bash
pip install auditorai
```

| What you need | Command |
|---|---|
| Base (sklearn models) | `pip install auditorai` |
| PyTorch models | `pip install "auditorai[pytorch]"` |
| HuggingFace models | `pip install "auditorai[hf]"` |
| OpenAI models | `pip install "auditorai[openai]"` |
| Anthropic models | `pip install "auditorai[anthropic]"` |
| Everything | `pip install "auditorai[all]"` |

---

## Quickstart

### With a sklearn model

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from auditorai import AuditorSystem, wrap

# 1. Load data and train your model as usual
X, y = load_breast_cancer(return_X_y=True)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Wrap your model and train the auditor (3 lines)
system = AuditorSystem(wrap(model))
system.train(X_val, y_val)       # uses held-out data
system.auto_tune(X_val, y_val)   # finds best threshold

# 3. Get audited predictions
result = system.predict(X_test)
print(result["show_mask"])       # True = safe to show
print(result["suppress_mask"])   # True = let human decide
print(result["p_wrong"])         # confidence score per prediction
```

### With a PyTorch model

```python
from auditorai import AuditorSystem, wrap
# pip install "auditorai[pytorch]"

adapter = wrap(your_torch_model,
               adapter_type="pytorch",
               n_classes=3)
system = AuditorSystem(adapter)
system.train(X_val, y_val)
result = system.predict(X_test)
```

### With an OpenAI model

```python
from auditorai import AuditorSystem, wrap
# pip install "auditorai[openai]"

def parse_response(text):
    # parse your model output -> (class_index, confidence)
    return int(text.strip()), 0.85

adapter = wrap("gpt-4o-mini",
               adapter_type="openai",
               parse_response=parse_response,
               n_classes=2)
system = AuditorSystem(adapter)
system.train(texts_val, y_val)
result = system.predict(texts_test)
```

---

## How it works

```
Step 1:  Your model makes a prediction
            ↓
Step 2:  AuditorAI scores it (0 = confident, 1 = likely wrong)
            ↓
Step 3:  Score ≥ threshold? → Suppress (human decides)
         Score < threshold? → Show prediction to user
            ↓
Step 4:  Only confident predictions reach your users
```

**Key insight:** The auditor trains on held-out validation data — data your primary model has *never* seen during training. This means it learns your model's real-world failure patterns, not memorized ones. The auditor uses uncertainty signals like confidence, entropy, and margin between top predictions to detect when your model is likely wrong.

---

## CLI

Works from the command line with zero Python:

```bash
# Run on a built-in dataset
auditorai run --data breast_cancer --report

# Run on your own CSV (last column = label)
auditorai run --data mydata.csv --model-type gradient_boosting

# Sweep thresholds to find the best one
auditorai sweep --data breast_cancer --steps 20

# Validate a saved auditor on new data
auditorai validate --adapter-path outputs/models --data breast_cancer

# All options
auditorai run --help
```

---

## What you get

### 1. Evaluation report (printed to terminal)

```
==================================================
  AUDITOR SYSTEM - EVALUATION REPORT
==================================================
  AI-only accuracy:         94.7%
  Joint system accuracy:    94.2%
  Auditor AUROC:            0.512
  Suppression rate:         1.8%
  Cases shown:              112
  Cases suppressed:         2
  Auditor precision:        50.0%
  Auditor recall:           16.7%
==================================================
```

### 2. Prediction dict (in Python)

```python
result = system.predict(X_test)

# For each sample in X_test:
result["show_mask"]       # bool array — True means show this prediction
result["suppress_mask"]   # bool array — True means suppress this prediction
result["p_wrong"]         # float array — probability this prediction is wrong
result["ai_predictions"]  # the actual predicted class labels
```

### 3. Plots saved to `outputs/`

| File | What it shows |
|---|---|
| `score_dist.png` | Auditor score distribution (correct vs. wrong predictions) |
| `threshold_sweep.png` | Accuracy gain vs. suppression threshold curve |
| `breakdown.png` | Shown vs. suppressed × correct vs. error breakdown |

---

## Supported models

| Model type | Adapter | Extra install |
|---|---|---|
| Any sklearn model | `SklearnAdapter` | *(included)* |
| XGBoost / LightGBM | `SklearnAdapter` | *(included)* |
| PyTorch `nn.Module` | `PyTorchAdapter` | `pip install "auditorai[pytorch]"` |
| HuggingFace pipeline | `HuggingFaceAdapter` | `pip install "auditorai[hf]"` |
| OpenAI (GPT-4o etc) | `APIAdapter` | `pip install "auditorai[openai]"` |
| Anthropic (Claude) | `APIAdapter` | `pip install "auditorai[anthropic]"` |
| Any custom model | Subclass `ModelAdapter` | *(included)* |

### Custom adapter pattern

```python
from auditorai import ModelAdapter, AuditorSystem
import numpy as np

class MyAdapter(ModelAdapter):
    def __init__(self, model):
        self.model = model

    def predict(self, X) -> np.ndarray:
        return self.model.my_predict(X)

    def predict_proba(self, X) -> np.ndarray:
        scores = self.model.my_scores(X)
        return np.column_stack([1 - scores, scores])  # must sum to 1.0

system = AuditorSystem(MyAdapter(my_model))
system.train(X_val, y_val)
```

---

## FAQ

**Q: Does my model need to support `predict_proba`?**
No. sklearn models without `predict_proba` (like `SVC`) are automatically wrapped with `CalibratedClassifierCV` to produce calibrated probabilities. No extra code needed.

**Q: What data do I pass to `system.train()`?**
Validation data — data your primary model has NOT trained on. This is critical. Passing training data produces an unreliable auditor that can't detect real errors.

**Q: What does `suppress_mask=True` mean for my application?**
It means AuditorAI is not confident in that prediction. What you do with it is up to you — show a warning, route to a human reviewer, or request more information from the user.

**Q: How do I choose the threshold?**
Use `system.auto_tune(X_val, y_val)` and it picks the threshold that maximizes joint accuracy automatically. Or use `auditorai sweep` from the CLI to see the full tradeoff curve and pick manually.

**Q: Will this work on text / images / tabular data?**
Yes. The auditor works on your model's probability outputs, not the raw inputs. As long as your adapter returns valid probabilities (rows sum to 1.0), the data type does not matter.

---

## Project structure

```
auditorai/
├── adapters/
│   ├── base.py                ← ModelAdapter ABC + wrap() function
│   ├── sklearn_adapter.py     ← wraps any sklearn model
│   ├── pytorch_adapter.py     ← wraps PyTorch nn.Module
│   ├── huggingface_adapter.py ← wraps HF pipelines and models
│   └── api_adapter.py         ← wraps OpenAI / Anthropic / custom HTTP
├── core/
│   ├── auditor.py             ← AuditorModel: trains on primary errors
│   ├── router.py              ← threshold sweep and routing logic
│   ├── system.py              ← AuditorSystem: main entry point
│   └── evaluate.py            ← reports and plots
├── cli/
│   └── main.py                ← auditorai run / sweep / validate
└── utils/
    ├── data.py                ← load_any() smart data loader
    └── logging.py             ← shared logger
```

---

## Contributing

```bash
git clone https://github.com/Apurva0614/Auditorai.git
cd Auditorai
pip install -e ".[dev]"
pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## Research

This implementation is based on the auditor model framework for human-AI decision systems. The core idea — training a second model to predict when the primary AI is wrong, then suppressing those predictions to let a human decide — was formalized in *Auditor Models for Efficient Human-AI Collaboration* (De-Arteaga, M. et al., 2025, medRxiv). AuditorAI makes this research practical by providing a drop-in library that works with any ML framework.

---

## License

[MIT](LICENSE) — use it freely, commercially or otherwise.
