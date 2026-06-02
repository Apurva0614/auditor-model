# Contributing to AuditorAI

First off, thank you for considering contributing to AuditorAI! It's people like you who make open source such a great community.

## Setting Up Your Development Environment

To set up the development environment, clone the repository and install the package with development dependencies:

```bash
git clone https://github.com/Apurva0614/Auditorai.git
cd Auditorai
pip install -e .[dev]
```

## Running Tests

We use `pytest` for unit testing. Make sure all tests pass before submitting a Pull Request:

```bash
pytest
```

To run with coverage reporting:

```bash
pytest --cov=auditorai tests/
```

## Pull Request Guidelines

To maintain a clean and reliable codebase, please adhere to these guidelines when preparing your PR:

1. **One Feature Per PR**: Keep your Pull Requests focused. Avoid bundling unrelated features or refactoring tasks.
2. **Tests Required**: Any new features or bug fixes must include corresponding tests to ensure coverage.
3. **Linear History**: We prefer clean, descriptive commits representing logical units of work.

## Code Style

We follow the standard Python styles and formatting conventions.
We use the following tools:
- **black** for code formatting.
- **ruff** for linting.

Before committing, run black on your changes:

```bash
black auditorai/ tests/
ruff check auditorai/ tests/
```
