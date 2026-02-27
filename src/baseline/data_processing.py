import pandas as pd
from rdkit import Chem


DEFAULT_QM9_COLS = ["mol_id", "smiles", "mu", "u0_atom", "homo", "lumo", "gap"]


def load_data(path="qm9.csv", n=20000):
    df = pd.read_csv(path)
    df = df.iloc[:n].copy()
    cols = DEFAULT_QM9_COLS
    return df[cols].copy()


def save_data(df, path="qm9_exp.csv"):
    df.to_csv(path, index=False)


def load_exp(path="qm9_exp.csv"):
    return pd.read_csv(path)


def check_null(df):
    return df.isnull().sum()


def mol(smiles):
    return Chem.MolFromSmiles(smiles)

