# auditor-model

## What this does

`auditor-model` trains a secondary "auditor" classifier that learns to predict when a primary AI model is likely to be wrong. When the auditor flags a prediction as unreliable, it suppresses the AI output and defers the decision to a human, improving overall system accuracy.

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

### Python API

```python
from sklearn.datasets import make_classification
from src.system import run_pipeline
from src.evaluate import run_full_evaluation

X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                            flip_y=0.08, random_state=42)

system, X_test, y_test = run_pipeline(X, y)
metrics = run_full_evaluation(system, X_test, y_test)
```

## How it works

The **PrimaryModel** is a calibrated scikit-learn classifier (random forest, gradient boosting, or logistic regression) that makes the initial prediction and outputs class probabilities. The **AuditorModel** is a gradient-boosting classifier trained on a held-out validation set to predict when the primary model is wrong. It uses four uncertainty signals derived from the primary's probability output: confidence (max probability), predicted class, Shannon entropy, and margin (gap between top-two probabilities). The **Router** applies a threshold to the auditor's P(wrong) score to decide per sample whether to show the AI prediction or suppress it and route to a human reviewer. The joint accuracy combines AI accuracy on shown cases with human accuracy on suppressed cases, yielding an improvement over deploying the AI alone.

## Project structure

```
auditor-model/
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── README.md                # This file
├── src/
│   ├── __init__.py
│   ├── utils.py             # Seed, data loading, splitting, model I/O, logging
│   ├── primary_model.py     # Calibrated primary classifier
│   ├── auditor_model.py     # Auditor: learns primary failure modes
│   ├── router.py            # Threshold-based routing and sweep
│   ├── system.py            # End-to-end orchestration
│   └── evaluate.py          # Reports and plots
├── tests/
│   ├── test_primary.py      # Unit tests for PrimaryModel
│   ├── test_auditor.py      # Unit tests for AuditorModel
│   └── test_system.py       # Unit tests for AuditorSystem
├── docs/
│   └── architecture.md      # Design rationale and extension guide
├── data/
│   ├── raw/                 # Raw input CSV files (gitignored)
│   └── processed/           # Processed data (gitignored)
├── outputs/
│   ├── models/              # Saved model files (gitignored)
│   ├── score_dist.png       # Auditor score distribution plot
│   ├── threshold_sweep.png  # Accuracy gain vs. threshold plot
│   └── breakdown.png        # Decision breakdown bar chart
└── notebooks/               # Jupyter notebooks (optional)
```

## CLI usage

```
python main.py [OPTIONS]

  --data PATH         CSV dataset path (last col = label).
                      Default: generate synthetic data.
  --model TYPE        random_forest | gradient_boosting | logistic
                      Default: random_forest
  --threshold FLOAT   Starting auditor threshold. Default: 0.5
  --human-acc FLOAT   Simulated human accuracy. Default: 0.72
  --no-tune           Skip auto_tune if set.
  --save-dir PATH     Model save directory. Default: outputs/models
  --output-dir PATH   Plot output directory. Default: outputs
```

## Tuning the threshold

The suppression threshold **tau** controls how aggressively the auditor flags predictions. A lower tau suppresses more predictions (higher recall, lower precision); a higher tau suppresses fewer (lower recall, higher precision).

`auto_tune` sweeps tau from 0.1 to 0.9 in 17 steps and selects the value that maximises joint accuracy gain. You can run this sweep explicitly:

```python
df = system.router_.sweep_thresholds(X_val, y_val, human_accuracy=0.80)
print(df.to_string(index=False))
```

## Running tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Research context

This system implements the auditor-model paradigm described in:

> Jabbour, S., et al. (2025). *An Auditor Model for AI-Assisted Decision-Making*. medRxiv. https://doi.org/10.1101/2023.04.03.23288014

The core idea — training a meta-model to predict primary AI errors and suppress unreliable cases before they reach end users — has been validated in clinical AI settings where undetected AI errors have high costs.

## License

MIT License. See [LICENSE](LICENSE) for details.
