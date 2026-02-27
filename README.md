# Structure-Aware Molecular Property Prediction on QM9

This repository contains baseline experiments for evaluating structure-awareness in molecular property prediction on the QM9 dataset.

The project investigates whether classical machine learning models (Linear Regression and Random Forest) can generalize under scaffold-based splits and preserve structural information in molecular representations.

This work forms the baseline stage of a NeurIPS-style workshop paper.

---

##  Project Overview

We study molecular property prediction on QM9 using Morgan fingerprints as molecular representations.

Our goals are:

1. Establish strong classical baselines
2. Evaluate generalization under scaffold splits
3. Analyze performance degradation under distribution shift
4. Assess ranking consistency of predictions

Future work will replace handcrafted fingerprints with learned graph neural network embeddings.

---

## Targets

We evaluate the following QM9 properties:

- `mu` (Dipole moment)
- `u0_atom` (Atomization energy)
- `homo` (HOMO energy)
- `lumo` (LUMO energy)
- `gap` (HOMO–LUMO gap)

---

## Experimental Setup

### Features
- Morgan fingerprints (radius=2, n_bits=2048)
- Generated via RDKit

### Models
- Linear Regression
- Random Forest Regressor

### Evaluation Metrics
- MAE
- RMSE
- R²
- Kendall τ (ranking consistency)
- Normalized MAE (MAE / target std)

### Data Splits
- Random split
- Scaffold split (Murcko scaffolds)

---

## Key Analyses

- Normalized MAE comparison
- Performance drop from random → scaffold split
- R² improvement of RF over Linear
- Ranking consistency under scaffold split
- Error vs molecule size

---

## Repository Structure

```
.
├── data_processing.py
├── featurization.py
├── random_split_experiments.py
├── random_split_plots.py
├── scaffold_split_experiments.py
├── scaffold_split_plots.py
├── Baseline_experiment.ipynb
└── README.md
```

---

## 🚀 Running the Experiments

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Required packages:

- pandas  
- numpy  
- scikit-learn  
- rdkit  
- matplotlib  
- scipy  

---

### 2️⃣ Run Random Split Experiments

```python
from random_split_experiments import run_random
```

---

### 3️⃣ Run Scaffold Split Experiments

```python
from scaffold_split_experiments import run_scaffold
```

---

##  Notes

- Default dataset size: 20,000 molecules  
- Random seed fixed for reproducibility  
- Code follows PEP8 formatting  
- Designed for clarity and modular experimentation  
