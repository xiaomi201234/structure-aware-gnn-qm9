import matplotlib.pyplot as plt

from random_split_experiments import r2_gain


def plot_norm_mae(norm_results):
    plt.figure(figsize=(8, 5))

    plt.plot(
        norm_results["Target"],
        norm_results["Linear_MAE"],
        marker="o",
        linewidth=2,
        color="#6BAED6",
        label="Linear",
    )

    plt.plot(
        norm_results["Target"],
        norm_results["RF_MAE"],
        marker="o",
        linewidth=2,
        color="#FB6A4A",
        label="Random Forest",
    )

    plt.ylabel("Normalized MAE")
    plt.title("Normalized Error Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_r2_improvement(df_results):
    improvement = r2_gain(df_results)
    fig = plt.figure(figsize=(8, 4))
    plt.bar(df_results["Target"], improvement, color="#6BAED6")
    plt.axhline(0, linestyle="--", color="gray", alpha=0.5)
    plt.ylabel("R² Improvement Ratio")
    plt.title("Random Split: RF Improvement over Linear (R²)")
    plt.tight_layout()
    return fig

