from data_processing import load_data, save_data, DEFAULT_QM9_COLS
from featurization import smiles_fp_all
from random_split_experiments import build_x, split_random, run_random, norm_mae, TARGETS as RANDOM_TARGETS
from scaffold_split_experiments import (
    add_scaffold,
    add_num_atoms,
    split_scaffold,
    run_scaffold,
    norm_scaffold,
    TARGETS as SCAFFOLD_TARGETS,
)


def run_random_baseline(
    path_or_df,
    n=20000,
    targets=None,
    test_size=0.2,
    random_state=42,
    rf_n=50,
    rf_depth=20,
    radius=2,
    n_bits=2048,
    save_exp_path=None,
):
    if targets is None:
        targets = RANDOM_TARGETS
    if hasattr(path_or_df, "columns"):
        df_exp = path_or_df[[c for c in DEFAULT_QM9_COLS if c in path_or_df.columns]].copy()
    else:
        df_exp = load_data(path_or_df, n=n)
    if save_exp_path:
        save_data(df_exp, save_exp_path)
    X = build_x(df_exp, radius=radius, n_bits=n_bits)
    df_results, (X_train, X_test, idx_train, idx_test) = run_random(
        X, df_exp, targets=targets, test_size=test_size, random_state=random_state, rf_n=rf_n, rf_depth=rf_depth
    )
    norm_results = norm_mae(df_results, df_exp)
    return df_results, norm_results, (X_train, X_test, idx_train, idx_test), df_exp


def run_scaffold_baseline(
    path_or_df,
    n=20000,
    targets=None,
    train_fraction=0.8,
    random_state=42,
    rf_n=50,
    rf_depth=20,
    radius=2,
    n_bits=2048,
    save_exp_path=None,
):
    if targets is None:
        targets = SCAFFOLD_TARGETS
    if hasattr(path_or_df, "columns"):
        df_exp = path_or_df[[c for c in DEFAULT_QM9_COLS if c in path_or_df.columns]].copy()
    else:
        df_exp = load_data(path_or_df, n=n)
    df_exp = add_scaffold(df_exp)
    df_exp = add_num_atoms(df_exp)
    if save_exp_path:
        save_data(df_exp, save_exp_path)
    X = build_x(df_exp, radius=radius, n_bits=n_bits)
    train_idx, test_idx = split_scaffold(df_exp, train_fraction=train_fraction, random_state=random_state)
    df_scaffold_summary, scaffold_results = run_scaffold(
        X, df_exp, train_idx, test_idx, targets=targets, rf_n=rf_n, rf_depth=rf_depth, random_state=random_state
    )
    norm_scaffold_df = norm_scaffold(df_scaffold_summary, df_exp)
    return df_scaffold_summary, scaffold_results, norm_scaffold_df, df_exp, (train_idx, test_idx)