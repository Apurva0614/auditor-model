# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-02

### Added
- Universal model adapter layer (`ModelAdapter` interface) supporting `scikit-learn`, `PyTorch`, `HuggingFace`, `OpenAI`, `Anthropic`, and custom wrappers.
- New `auditorai` CLI with `run`, `sweep`, and `validate` subcommands.
- `AuditorDriftDetector` class in `auditorai/monitor.py` to monitor and detect drift in the auditor's prediction suppression rates.
- Support for `feature_fn` custom feature mapping callbacks in `AuditorModel`.
- Automated GitHub Actions CI workflow for multiple Python versions.
- Automated release workflow for publishing to PyPI using OIDC trusted publishing.
- Example notebooks demonstrating exploration, training, and evaluation.

## [0.1.0] - 2026-06-01

### Added
- Initial release of the `auditor-model` prototype.
- Basic prediction suppression logic for human-AI decision systems.
- Core routing and evaluation metrics.
