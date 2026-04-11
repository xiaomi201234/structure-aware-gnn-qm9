# Scaffold Distribution Shift and Structural Generalization in Graph Neural Networks

This repository contains the code for our project on **structural generalization in molecular property prediction** under scaffold-induced distribution shift.

We study how different graph neural networks generalize to **structurally novel molecules** on the QM9 dataset, compared with classical baselines. In addition to predictive accuracy, we analyze performance degradation, representation stability, scaffold separability, and fine-grained structural sensitivity under non-IID settings.

This project was developed in the style of a workshop paper and focuses on understanding **why** some models are more robust than others under scaffold shift.

---

## Project Overview

We investigate molecular property prediction on **QM9** under two evaluation settings:

1. **Random split** for approximate IID evaluation  
2. **Scaffold split** based on Murcko scaffolds for structure-aware, non-IID evaluation

Our goals are:

1. Compare classical baselines and representative GNN architectures  
2. Measure performance degradation under scaffold-induced distribution shift  
3. Analyze whether representation stability aligns with generalization  
4. Study scaffold separability in learned embedding space  
5. Examine model sensitivity to molecular size, topology, scaffold size, and scaffold frequency

---

## Targets

We evaluate the following seven QM9 properties:

- `mu` — Dipole moment
- `u0_atom` — Atomization energy
- `homo` — HOMO energy
- `lumo` — LUMO energy
- `gap` — HOMO–LUMO gap
- `alpha` — Polarizability
- `cv` — Heat capacity

Correlated targets such as `U0 / U / H / G` and related atomization variants are excluded.

---

## Models

### Classical Baselines
- Linear Regression
- Random Forest

### GNN Models
- GCN
- GraphSAGE
- GIN
- GAT
- MPNN
- Graph Transformer (GT)

All GNNs are trained under a unified graph-level prediction framework with stacked graph layers, global mean pooling, and an MLP regression head.

---

## Molecular Representation

### Baselines
- Morgan fingerprints
- `radius = 2`
- `n_bits = 2048`

### GNNs
Molecules are represented as graphs where:

- atoms are **nodes**
- bonds are **edges**
- each node has an **8-dimensional atomic feature vector**
- each edge has a **3-dimensional bond feature vector**

SMILES strings are parsed with **RDKit** and converted into **PyTorch Geometric Data objects**.

---

## Evaluation Metrics

### Standard Prediction Metrics
- MAE
- RMSE
- R²
- Kendall τ

### Structural Generalization Metrics
- **SDI** — Scaffold Degradation Index  
  Measures relative performance degradation under scaffold split:
  
  `SDI = (MAE_scaffold - MAE_random) / MAE_random`

- **RSI** — R² Stability Index  
  Measures retained explanatory power under scaffold shift:
  
  `RSI = R²_scaffold / R²_random`

- **SSR** — Scaffold Separation Ratio  
  Measures scaffold separability in embedding space:
  
  `SSR = E[d_inter] / E[d_intra]`

- **Sensitivity**  
  Relative scaffold-vs-random error ratio:
  
  `Sensitivity = MAE_scaffold / MAE_random`

We estimate aggregated means and **95% confidence intervals** using bootstrap resampling.

---

## Experimental Setup

### Dataset
- QM9
- 70,000 molecules used in experiments

### Splits
- **Random split**
- **Scaffold split** using Murcko scaffolds  
  No scaffold overlap between training, validation, and test sets

### GNN Training
- Single-task regression for each property
- Optimizer: Adam
- Loss: MSE
- Learning rate: `1e-3`
- Batch size: `64`
- Max epochs: `300`
- Best checkpoint selected by validation `R²`

### Random Forest
- `n_estimators = 50`
- `max_depth = 20`

---

## Key Analyses

This repository supports the following analyses:

### 1. Scaffold Split Performance Analysis
- SDI across models and properties
- RSI across models and properties
- Comparison between baselines and GNNs

### 2. Representation Structure Analysis
- Graph embedding extraction
- PCA visualization of test-set embeddings
- Scaffold Separation Ratio (SSR)

### 3. Fine-Grained Structural Analysis
- Error vs atom count
- Error vs ring count
- Error vs scaffold size
- Error vs scaffold frequency
- Rare-scaffold failure analysis

### 4. Model-by-Property Verification
- Heatmap analysis to verify that property-level patterns are not artifacts of aggregation across model classes

---

## Main Findings

Our experiments show that:

- All models degrade under scaffold split relative to random split
- GCN shows the strongest degradation under scaffold shift
- GIN, MPNN, and Graph Transformer are generally more robust
- `cv` and `alpha` are the most scaffold-sensitive targets
- `mu` and `u0_atom` are comparatively more stable
- Representation stability does **not** perfectly explain scaffold generalization
- Rare scaffolds and increasing structural complexity remain challenging

---

## Repository Structure

```text
.
├── Baseline/
│   ├── data_processing.py
│   ├── featurization.py
│   ├── pipeline.py
│   ├── random_split_experiments.py
│   ├── random_split_plots.py
│   ├── scaffold_split_experiments.py
│   ├── scaffold_split_plots.py
│
├── GNN/
│   ├── split.py
│   ├── train.py
│
├── model/
│   ├── gcn.py
│   ├── graphsage.py
│   ├── gin.py
│   ├── gat.py
│   ├── mpnn.py
│   ├── gt.py
│
├── Representation Learning/
│   ├── graph_build.py
│   ├── embeddings.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── plots.py
│   ├── scaffold_utils.py
│   ├── sensitivity.py
│
├── Baseline_experiment.ipynb
└── README.md
