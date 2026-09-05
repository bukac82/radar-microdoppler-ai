# Dataset Module

This module handles dataset generation and loading for the Radar Micro-Doppler AI project.

## Files

| File | Description |
|------|-------------|
| `generate_dataset.py` | Generates the base helicopter micro-Doppler dataset (100k samples, X-band 10 GHz, fixed elevation 10°) |
| `generate_dataset_extended.py` | Generates the extended dataset with varied radar frequency (8–12 GHz), elevation angle (0–45°), and bulk target velocity (±50 m/s) |
| `create_docx.py` | Creates the README docx documentation |
| `loader.py` | Utilities for loading, chunking, and splitting the CSVs |

## Dataset Overview

- **Classes**: 2-blade (Bell UH-1), 3-blade (Gazelle), 4-blade (Apache / UH-60)
- **Samples**: 100,000 per dataset
- **Signal**: Complex IQ (I + jQ), 500 samples @ 1 kHz → 0.5s window
- **Label**: `num_blades`

## Usage

```python
from Dataset.loader import load_dataset, get_train_test_split

df = load_dataset("helicopter_microdoppler_dataset.csv", nrows=10000)
X_train, X_test, y_train, y_test = get_train_test_split(df)
```
