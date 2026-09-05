# Contributing to Radar Micro-Doppler AI

Thank you for your interest in contributing! This project welcomes contributions
of all kinds — new models, bug fixes, additional datasets, documentation, and experiments.

---

## 🚀 Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/radar-microdoppler-ai.git
cd radar-microdoppler-ai
```

### 2. Set Up Environment

```bash
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
# .\.venv\Scripts\activate          # Windows

pip install -e ".[all,dev]"
```

### 3. Get the Data

```bash
python scripts/download_data.py --source generate
# or point to existing CSVs in the project root
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

Each module lives in its own directory. When adding to a module:
- Follow existing code style (docstrings, type hints, `if __name__ == "__main__"` smoke tests)
- Add or update the corresponding `README.md` inside the module
- Add unit tests in `tests/`

---

## 🤝 Types of Contributions

### New Models
- Place new classical models in `Classical_ML/train.py`
- New DL architectures go in `Deep_Learning/` as `<arch_name>.py`
- New quantum circuits in `Quantum_ML/circuits.py`

### New Datasets / Signals
- Extend `Dataset/generate_dataset_extended.py` for new physical parameters
- Add drone / bird / UAV signatures as new classes

### New Robustness Tests
- Add to `Robustness_Testing/` (e.g., multipath, clutter, jamming)

### New Explainability Methods
- Add to `Explainability/` (e.g., LIME, Integrated Gradients)

---

## 📝 Code Style

- **Python 3.10+** with type hints where possible
- Docstrings: Google style (`Args:`, `Returns:`, `Raises:`)
- Line length: 100 characters
- Formatting: `ruff format .` before committing

---

## 🧪 Testing Guidelines

- Every new function should have at least one test in `tests/`
- Tests should be fast (no loading of full 2 GB CSV — use `_make_dummy_df` helpers)
- Run `pytest tests/ --tb=short` before opening a PR

---

## 📬 Submitting a Pull Request

1. Create a branch: `git checkout -b feature/my-new-model`
2. Commit with clear messages: `git commit -m "feat(DL): add ConvLSTM architecture"`
3. Push and open a PR against `main`
4. Fill in the PR template — describe the change, results achieved, and how to test

---

## 🐛 Reporting Issues

Open an issue with:
- A minimal reproducible example
- Your Python version and OS
- Relevant error message / traceback

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the **MIT License**.
