---
license: mit
language:
- en
tags:
- radar
- signal-processing
- micro-doppler
- helicopter
- classification
- synthetic
- iq-data
- time-series
pretty_name: Micro-Doppler Signatures (Helicopter)
size_categories:
- 100K<n<1M
task_categories:
- tabular-classification
- audio-classification
---

# Micro-Doppler Signatures — Helicopter Classification Dataset

Synthetic IQ-sampled radar returns from three helicopter types, generated using a physics-based sinc micro-Doppler scattering model. Designed for benchmarking ML classifiers on rotating-blade target identification.

## Dataset Description

### What's in it

Two CSV files, each with 100,000 samples:

| File | Contents |
|------|----------|
| `helicopter_microdoppler_dataset.csv` | Baseline — fixed radar geometry, SNR ∈ [5, 25] dB |
| `helicopter_microdoppler_extended_dataset.csv` | Extended — variable radar frequency (8–12 GHz), elevation angle (0–45°), bulk target velocity (±50 m/s) |

Each row is one 0.5-second radar observation window at 1 kHz sampling rate (500 complex IQ samples per window), flattened as real and imaginary columns.

### Target classes

| Label | Helicopter | Blades | Rotor RPM | Blade length |
|-------|-----------|--------|-----------|--------------|
| `2` | Bell UH-1 Iroquois (Huey) | 2 | 300–350 | 6.5–7.5 m |
| `3` | Aérospatiale Gazelle | 3 | 360–400 | 4.5–5.5 m |
| `4` | Boeing AH-64 Apache / UH-60 Black Hawk | 4 | 250–300 | 7.0–8.5 m |

### Signal model

The micro-Doppler return from the $k$-th blade is modelled as:

$$s_k(t) = L \cdot \text{sinc}\!\left(\frac{2L}{\lambda}\cos\phi_k\cos\beta\right) \exp\!\left(j\frac{4\pi L}{\lambda}\cos\phi_k\cos\beta\right)$$

where $\phi_k(t) = \omega t + \theta_0 + \frac{2\pi k}{N_b}$ is the instantaneous blade phase, $L$ is blade half-length, $\lambda$ is radar wavelength, and $\beta$ is the elevation angle. AWGN is added to reach the target SNR.

## Column Schema

```
label          — int {2, 3, 4}         helicopter class (number of main rotor blades)
snr_db         — float                 signal-to-noise ratio of this sample
n_blades       — int                   number of rotor blades
rpm            — float                 rotor revolutions per minute
blade_length_m — float                 blade half-length in metres
I_0 … I_499    — float                 in-phase (real) IQ samples
Q_0 … Q_499    — float                 quadrature (imaginary) IQ samples
```

Extended dataset additionally includes:
```
radar_freq_ghz — float                 radar carrier frequency (8–12 GHz)
elevation_deg  — float                 target elevation angle (0–45°)
velocity_ms    — float                 bulk target radial velocity (−50 to +50 m/s)
```

## Intended Use

- Benchmarking classical and deep learning classifiers on radar micro-Doppler data
- Evaluating robustness to noise (SNR sweep experiments)
- Research into quantum kernel methods on signal classification tasks
- Open-set recognition and out-of-distribution detection studies

## Limitations

- Synthetic data only — real radar returns include ground clutter, multipath, and hardware-specific artefacts not modelled here
- Three helicopter classes only — does not cover fixed-wing aircraft, drones, or birds
- Monostatic radar geometry assumed

## Related Repository

Code, notebooks, and full experimental pipeline:  
**[bukac82/radar-microdoppler-ai](https://github.com/bukac82/radar-microdoppler-ai)**

## Citation

If you use this dataset in your research, please cite the dataset/software repository and/or the relevant papers below.

### Dataset & Software Repository
```bibtex
@software{agnihotri2026microdoppler,
  author    = {Agnihotri, Vikas},
  title     = {Radar Micro-Doppler AI: End-to-End Helicopter Classification},
  year      = {2026},
  url       = {https://github.com/bukac82/radar-microdoppler-ai}
}
```

### Related Papers

**Quantum ML on NISQ Hardware:**
```bibtex
@article{agnihotri2026quantum,
  author    = {Agnihotri, Vikas and Kaur, Jasleen and Kaushik, Sarvagya},
  title     = {Practical Evaluation of Quantum Kernel Methods for Radar
               Micro-Doppler Classification on Noisy Intermediate-Scale
               Quantum ({NISQ}) Hardware},
  journal   = {arXiv preprint},
  volume    = {arXiv:2601.22194},
  year      = {2026},
  url       = {https://arxiv.org/abs/2601.22194}
}
```

**Radar-Based ATR Framework (foundational SVM/signal model):**
```bibtex
@article{agnihotri2020radar,
  author    = {Agnihotri, Vikas and Sabharwal, Munish},
  title     = {An Automatic Radar Based Aerial Target Recognition Framework},
  journal   = {Journal of Interdisciplinary Mathematics},
  volume    = {23},
  number    = {2},
  pages     = {321--333},
  year      = {2020},
  doi       = {10.1080/09720502.2020.1737377},
  url       = {https://doi.org/10.1080/09720502.2020.1737377}
}
```

**Frequency Effects on Micro-Doppler (underpins extended dataset design):**
```bibtex
@inproceedings{agnihotri2019frequency,
  author    = {Agnihotri, Vikas and Sabharwal, Munish and Goyal, Vinay},
  title     = {Effect of Frequency on Micro-Doppler Signatures of a Helicopter},
  booktitle = {2019 International Conference on Advances in Big Data,
               Computing and Data Communication Systems (icABCD)},
  year      = {2019},
  doi       = {10.1109/ICABCD.2019.8851024},
  url       = {https://doi.org/10.1109/ICABCD.2019.8851024}
}
```

## License

MIT — free to use for research and commercial purposes with attribution.

