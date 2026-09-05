# 📁 Dataset Module

This module handles dataset management, loading, and generation for the Radar Micro-Doppler AI project.

## 🤗 HuggingFace Dataset

The complete dataset (~4 GB total) is hosted on the Hugging Face Hub:

👉 **[huggingface.co/datasets/bukac82/MicroDopplerSignatures](https://huggingface.co/datasets/bukac82/MicroDopplerSignatures)**

### ⬇️ How to Obtain the Dataset

#### Option 1 — Download Script (Recommended)
```bash
# Download from HuggingFace Hub directly to project root:
python scripts/download_data.py

# Or if access is gated / requires authentication:
python scripts/download_data.py --token <YOUR_HF_TOKEN>
```

#### Option 2 — HuggingFace CLI
```bash
huggingface-cli download bukac82/MicroDopplerSignatures \
    --repo-type dataset --local-dir .
```

#### Option 3 — Python `datasets` Library
```python
from datasets import load_dataset

# Stream or load from HuggingFace Hub
ds = load_dataset("bukac82/MicroDopplerSignatures")
```

#### Option 4 — Generate Locally (CPU, ~20–40 min)
```bash
python scripts/download_data.py --source generate
# or run generation scripts directly:
python Dataset/generate_dataset.py
python Dataset/generate_dataset_extended.py
```

---

## 📄 Files in this Directory

| File | Description |
|------|-------------|
| `loader.py` | Utilities for loading, chunking, and train/test splitting the CSVs (`load_dataset`, `get_iq_matrix`, `get_train_test_split`) |
| `generate_dataset.py` | Physics-based generator for the base dataset (100k samples, X-band 10 GHz, 10° elevation) |
| `generate_dataset_extended.py` | Physics-based generator for the extended dataset (100k samples, 8–12 GHz, 0–45° elevation, ±50 m/s velocity) |

---

## 📊 Dataset Overview

Two physics-accurate synthetic datasets generated from the **sinc-based micro-Doppler radar model**:

| Dataset | Samples | Features | Size | Target Classes |
|---------|---------|----------|------|----------------|
| `helicopter_microdoppler_dataset.csv` | 100,000 | 500 complex IQ @ 1 kHz (0.5s window) | ~2 GB | 2, 3, 4 blades |
| `helicopter_microdoppler_extended_dataset.csv` | 100,000 | 500 complex IQ + freq + elevation + bulk velocity | ~2 GB | 2, 3, 4 blades |

### Target Classes

- **Class 2 (2-blade)**: Bell UH-1 Iroquois (Huey) — 300–350 RPM, 6.5–7.5 m blade
- **Class 3 (3-blade)**: Aérospatiale SA 341 Gazelle — 360–400 RPM, 4.5–5.5 m blade
- **Class 4 (4-blade)**: Boeing AH-64 Apache / Sikorsky UH-60 Black Hawk — 250–300 RPM, 7.0–8.5 m blade

---

## 💻 Python Usage

```python
from Dataset.loader import load_dataset, get_train_test_split

# Load base or extended dataset
df = load_dataset("helicopter_microdoppler_dataset.csv", nrows=10000)

# Extract IQ matrices and split into train/test
X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2)

print(f"X_train shape: {X_train.shape}")  # (8000, 500) complex128
print(f"y_train shape: {y_train.shape}")  # (8000,)
```

