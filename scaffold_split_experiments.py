import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import kendalltau


TARGETS = ["mu", "u0_atom", "homo", "lumo", "gap"]


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def add_scaffold(df_exp):
    df = df_exp.copy()
    df["scaffold"] = df["smiles"].apply(get_scaffold)
    return df


def split_scaffold(df_with_scaffold, train_fraction=0.8, random_state=42):
    scaffold_groups = df_with_scaffold.groupby("scaffold").indices
    scaffolds = list(scaffold_groups.keys())

    rng = np.random.RandomState(random_state)
    rng.shuffle(scaffolds)

    train_idx = []
    test_idx = []

    n_total = len(df_with_scaffold)
    n_train_target = int(train_fraction * n_total)

    for scaffold in scaffolds:
        indices = scaffold_groups[scaffold]
        if len(train_idx) + len(indices) <= n_train_target:
            train_idx.extend(indices)
        else:
            test_idx.extend(indices)

    return np.array(train_idx), np.array(test_idx)


def add_num_atoms(df_exp):
    df = df_exp.copy()
    df["num_atoms"] = df["smiles"].apply(
        lambda s: Chem.MolFromSmiles(s).GetNumAtoms()
    )
    return df


def run_scaffold(X, df_exp_with_scaffold_and_atoms, train_idx, test_idx, targets=None, rf_n=50, rf_depth=20, random_state=42):
    if targets is None:
        targets = TARGETS

    scaffold_results = {}
    summary_results = []

    X_train = X[train_idx]
    X_test = X[test_idx]

    for t in targets:
        y = df_exp_with_scaffold_and_atoms[t].values
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaffold_results[t] = {}

        lin = LinearRegression()
        lin.fit(X_train, y_train)
        y_pred_lin = lin.predict(X_test)

        lin_mae = mean_absolute_error(y_test, y_pred_lin)
        lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
        lin_r2 = r2_score(y_test, y_pred_lin)
        lin_tau, _ = kendalltau(y_test, y_pred_lin)

        scaffold_results[t]["Linear"] = {
            "y_test": y_test,
            "y_pred": y_pred_lin,
            "errors": np.abs(y_test - y_pred_lin),
            "num_atoms": df_exp_with_scaffold_and_atoms.iloc[test_idx][
                "num_atoms"
            ].values,
        }

        rf = RandomForestRegressor(
            n_estimators=rf_n,
            max_depth=rf_depth,
            n_jobs=-1,
            random_state=random_state,
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        rf_mae = mean_absolute_error(y_test, y_pred_rf)
        rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        rf_r2 = r2_score(y_test, y_pred_rf)
        rf_tau, _ = kendalltau(y_test, y_pred_rf)

        scaffold_results[t]["RF"] = {
            "y_test": y_test,
            "y_pred": y_pred_rf,
            "errors": np.abs(y_test - y_pred_rf),
            "num_atoms": df_exp_with_scaffold_and_atoms.iloc[test_idx][
                "num_atoms"
            ].values,
        }

        summary_results.append(
            [
                t,
                lin_mae,
                lin_rmse,
                lin_r2,
                lin_tau,
                rf_mae,
                rf_rmse,
                rf_r2,
                rf_tau,
            ]
        )

    df_scaffold_summary = pd.DataFrame(
        summary_results,
        columns=[
            "Target",
            "Linear_MAE",
            "Linear_RMSE",
            "Linear_R2",
            "Linear_Kendall",
            "RF_MAE",
            "RF_RMSE",
            "RF_R2",
            "RF_Kendall",
        ],
    )

    return df_scaffold_summary, scaffold_results


def norm_scaffold(df_scaffold_summary, df_exp):
    norm = df_scaffold_summary.copy()

    for t in norm["Target"]:
        std = df_exp[t].std()
        norm.loc[norm["Target"] == t, "Linear_MAE"] /= std
        norm.loc[norm["Target"] == t, "RF_MAE"] /= std

    return norm


def r2_gain(df_scaffold_summary):
    return (df_scaffold_summary["RF_R2"] - df_scaffold_summary["Linear_R2"]) / (
        df_scaffold_summary["Linear_R2"]
    )
