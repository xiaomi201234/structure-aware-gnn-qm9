import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import kendalltau

from featurization import smiles_fp_all


TARGETS = ["mu", "u0_atom", "homo", "lumo", "gap"]


def build_x(df_exp, radius=2, n_bits=2048):
    return smiles_fp_all(df_exp["smiles"], radius=radius, n_bits=n_bits)


def split_random(X, test_size=0.2, random_state=42):
    X_train, X_test, idx_train, idx_test = train_test_split(
        X, np.arange(len(X)), test_size=test_size, random_state=random_state
    )
    return X_train, X_test, idx_train, idx_test


def run_random(X, df_exp, targets=None, test_size=0.2, random_state=42, rf_n=50, rf_depth=20):
    if targets is None:
        targets = TARGETS

    X_train, X_test, idx_train, idx_test = split_random(
        X, test_size=test_size, random_state=random_state
    )

    results = []

    for t in targets:
        y = df_exp[t].values
        y_train = y[idx_train]
        y_test = y[idx_test]

        lin = LinearRegression()
        lin.fit(X_train, y_train)
        y_pred_lin = lin.predict(X_test)

        lin_mae = mean_absolute_error(y_test, y_pred_lin)
        lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
        lin_r2 = r2_score(y_test, y_pred_lin)
        lin_tau, _ = kendalltau(y_test, y_pred_lin)

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

        results.append(
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

    df_results = pd.DataFrame(
        results,
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

    return df_results, (X_train, X_test, idx_train, idx_test)


def norm_mae(df_results, df_exp):
    norm_results = df_results.copy()

    for t in norm_results["Target"]:
        std = df_exp[t].std()
        norm_results.loc[norm_results["Target"] == t, "Linear_MAE"] /= std
        norm_results.loc[norm_results["Target"] == t, "RF_MAE"] /= std

    return norm_results

