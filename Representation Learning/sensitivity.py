import pandas as pd
import numpy as np


def structure_sensitivity(df, mae_random_col="mae_random", mae_scaffold_col="mae_scaffold"):
    out = df.copy()
    out["sensitivity"] = out[mae_scaffold_col] / out[mae_random_col]
    return out


def sensitivity_by_property(df, property_col="property", sensitivity_col="sensitivity", order=None):
    s = df.groupby(property_col)[sensitivity_col].mean()
    if order is not None:
        s = s.reindex(order)
    return s


def error_by_groups(df, group_col, model_col="model", error_col="error"):
    return df.groupby([model_col, group_col])[error_col].mean().reset_index()


def rank_matrix_from_error_matrix(error_matrix):
    rank_matrix = np.zeros_like(error_matrix, dtype=float)
    for j in range(error_matrix.shape[1]):
        order = np.argsort(error_matrix[:, j])
        rank_matrix[order, j] = np.arange(1, len(order) + 1)
    return rank_matrix
