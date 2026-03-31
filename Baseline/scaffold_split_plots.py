import numpy as np
import matplotlib.pyplot as plt

from scaffold_split_experiments import r2_gain, r2_drop


def plot_scaffold_norm(norm_scaffold):
    targets = norm_scaffold["Target"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        targets,
        norm_scaffold["Linear_MAE"],
        marker="o",
        linewidth=2,
        color="#6BAED6",
        label="Linear",
    )

    plt.plot(
        targets,
        norm_scaffold["RF_MAE"],
        marker="o",
        linewidth=2,
        color="#FB6A4A",
        label="Random Forest",
    )

    plt.ylabel("Normalized MAE")
    plt.title("Scaffold Split: Normalized Error Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_scaffold_r2(df_scaffold_summary):
    vals = r2_gain(df_scaffold_summary)

    plt.figure(figsize=(8, 4))

    plt.bar(df_scaffold_summary["Target"], vals, color="#6BAED6")
    plt.axhline(0, linestyle="--", color="gray", alpha=0.5)
    plt.ylabel("R² Improvement Ratio")
    plt.title("Scaffold Split: RF Improvement over Linear (R²)")
    plt.tight_layout()

    return plt.gcf()


def plot_scaffold_k(df_scaffold_summary):
    plt.figure(figsize=(8, 5))

    plt.plot(
        df_scaffold_summary["Target"],
        df_scaffold_summary["Linear_Kendall"],
        marker="o",
        linewidth=2,
        color="#6BAED6",
        label="Linear",
    )

    plt.plot(
        df_scaffold_summary["Target"],
        df_scaffold_summary["RF_Kendall"],
        marker="o",
        linewidth=2,
        color="#FB6A4A",
        label="Random Forest",
    )

    plt.ylabel("Kendall τ")
    plt.title("Scaffold Split: Ranking Consistency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_r2_drop(df_random_results, df_scaffold_summary):
    drop = r2_drop(df_random_results, df_scaffold_summary)
    targets = df_scaffold_summary["Target"]
    fig = plt.figure(figsize=(8, 4))
    plt.bar(targets, drop, color="#FB6A4A")
    plt.axhline(0, linestyle="--", color="gray", alpha=0.5)
    plt.ylabel("R² Drop (Random → Scaffold)")
    plt.title("Scaffold Split: R² Degradation vs Random")
    plt.tight_layout()
    return fig


def plot_errors_by_atoms(scaffold_results, targets=None, model="RF", figsize=(12, 6)):
    if targets is None:
        targets = list(scaffold_results.keys())
    n = len(targets)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.atleast_2d(axes)
    for i, t in enumerate(targets):
        ax = axes.flat[i]
        data = scaffold_results[t][model]
        ax.scatter(data["num_atoms"], data["errors"], alpha=0.5, s=10)
        ax.set_xlabel("Number of atoms")
        ax.set_ylabel("Error")
        ax.set_title(t)
    for j in range(i + 1, axes.size):
        axes.flat[j].set_visible(False)
    plt.tight_layout()
    return fig

