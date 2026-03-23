# Smart Relay — Fault Detection & Classification

AI-based relay prototype for detection and classification of electrical faults
in three-phase systems, implemented on NVIDIA Jetson Nano.

## Project structure

| Folder           | Purpose                                              |
|------------------|------------------------------------------------------|
| `data/raw/`      | Original CSVs from DIgSILENT — read-only             |
| `data/processed/`| Labeled feature sets + fitted scaler                 |
| `data/splits/`   | Stratified train / val / test partitions             |
| `detection/`     | Binary fault detection module (RF · SVM · MLP)       |
| `classification/`| Multiclass fault classification module               |
| `utils/`         | Shared preprocessing, plotting, and report scripts   |
| `notebooks/`     | EDA only — no production logic                       |
| `tests/`         | Unit tests                                           |

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place raw CSVs
cp your_files/*.csv data/raw/

# 3. Preprocess and split
python utils/preprocess.py
python utils/split.py

# 4. Train detection module
python detection/train.py

# 5. Train classification module
python classification/train.py
```

## IEEE standards
- **IEC 60255** — Measuring relays and protection equipment
- **IEEE C37.100** — Dependability (Recall) and Security (Specificity)
- **IEC 61850** — Communication networks and systems in substations

## Team
- Santiago Castro Sierra
- Jesús Manuel Carmona Acuña  
- Alfredo Alberto Arraut Navarro

**Advisors:** Mauricio Restrepo Restrepo · Rafael Castillo Sierra  
**Institution:** Universidad del Norte — 2026-10
