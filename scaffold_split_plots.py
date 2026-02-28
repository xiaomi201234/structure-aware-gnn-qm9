import matplotlib.pyplot as plt

from scaffold_split_experiments import r2_gain


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

