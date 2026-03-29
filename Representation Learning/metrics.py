import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "Property", "Model",
    "MAE_random", "R2_random", "Kendall_random",
    "MAE_scaffold", "R2_scaffold", "Kendall_scaffold"
]


def compute_sdi(df):
    out = df.copy()
    out["SDI"] = (out["MAE_scaffold"] - out["MAE_random"]) / out["MAE_random"]
    return out


def average_sdi(df, by="Model"):
    return df.groupby(by)["SDI"].mean().sort_values(ascending=(by == "Property"))


def compute_rsi(df):
    out = df.copy()
    out["RSI"] = out["R2_scaffold"] / out["R2_random"]
    return out


def bootstrap_sdi(y_random_true, y_random_pred, y_scaffold_true, y_scaffold_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_random_true)
    sdi_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        mae_r = np.mean(np.abs(y_random_true[idx] - y_random_pred[idx]))
        mae_s = np.mean(np.abs(y_scaffold_true[idx] - y_scaffold_pred[idx]))
        sdi = (mae_s - mae_r) / mae_r if mae_r > 0 else 0.0
        sdi_samples.append(sdi)
    sdi_samples = np.array(sdi_samples)
    return float(np.mean(sdi_samples)), float(np.percentile(sdi_samples, 2.5)), float(np.percentile(sdi_samples, 97.5))


def bootstrap_sdi_by_property(df, n_boot=2000):
    boot_prop = {}
    for prop in df["Property"].unique():
        vals = df[df["Property"] == prop]["SDI"].values
        samples = []
        for _ in range(n_boot):
            resample = np.random.choice(vals, size=len(vals), replace=True)
            samples.append(np.mean(resample))
        samples = np.array(samples)
        boot_prop[prop] = [np.mean(samples), np.percentile(samples, 2.5), np.percentile(samples, 97.5)]
    out = pd.DataFrame(boot_prop).T
    out.columns = ["mean", "lower", "upper"]
    out["err_low"] = out["mean"] - out["lower"]
    out["err_high"] = out["upper"] - out["mean"]
    return out


def sdi_pivot(df):
    return df.pivot(index="Property", columns="Model", values="SDI")


def sdi_correlation(df):
    return df.pivot(index="Property", columns="Model", values="SDI").corr()


def build_bootstrap_df(summary_dict, index_name="Model", order=None):
    rows = []
    for key, (mean, lower, upper) in summary_dict.items():
        rows.append({index_name: key, "mean": mean, "lower": lower, "upper": upper})
    out = pd.DataFrame(rows)
    out["err_low"] = out["mean"] - out["lower"]
    out["err_high"] = out["upper"] - out["mean"]
    if order is not None:
        out = out.set_index(index_name).reindex(order).reset_index()
    return out
