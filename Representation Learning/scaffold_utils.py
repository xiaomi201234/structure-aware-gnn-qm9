from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import pandas as pd


def _scaffold_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def add_scaffold_columns(df, smiles_col="smiles", rare=5, medium=20):
    out = df.copy()
    out["scaffold"] = out[smiles_col].apply(_scaffold_smiles)
    freq = out.groupby("scaffold").size()
    out["scaffold_freq"] = out["scaffold"].map(freq)
    def bucket(f):
        if f <= rare:
            return "Rare Scaffold"
        if f <= medium:
            return "Medium Scaffold"
        return "Frequent Scaffold"
    out["scaffold_bucket"] = out["scaffold_freq"].apply(bucket)
    return out


def add_scaffold_size(df, scaffold_col="scaffold"):
    out = df.copy()
    sz = out.groupby(scaffold_col).size()
    out["scaffold_size"] = out[scaffold_col].map(sz)
    return out


def add_ring_count(df, smiles_col="smiles"):
    def rings(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return np.nan if mol is None else mol.GetRingInfo().NumRings()
    out = df.copy()
    out["ring_count"] = out[smiles_col].apply(rings)
    return out


def scaffold_train_test_split(df, scaffold_col="scaffold", train_frac=0.8, seed=42):
    groups = df.groupby(scaffold_col).indices
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_total = len(df)
    n_train_target = int(train_frac * n_total)
    train_idx, test_idx = [], []
    for sc in keys:
        idx = groups[sc]
        if len(train_idx) + len(idx) <= n_train_target:
            train_idx.extend(idx)
        else:
            test_idx.extend(idx)
    return np.array(train_idx), np.array(test_idx)


def add_error_column(df, y_pred_col="y_pred", y_true_col="y_true"):
    out = df.copy()
    out["error"] = np.abs(out[y_pred_col] - out[y_true_col])
    return out


def rare_scaffold_analysis(df, model_col="model", bucket_col="scaffold_bucket", error_col="error"):
    mean_err = df.groupby([model_col, bucket_col])[error_col].mean().reset_index()
    pivot = mean_err.pivot(index=model_col, columns=bucket_col, values=error_col)
    if "Rare Scaffold" in pivot.columns and "Frequent Scaffold" in pivot.columns:
        pivot["rare_degradation"] = pivot["Rare Scaffold"] - pivot["Frequent Scaffold"]
    return mean_err, pivot
