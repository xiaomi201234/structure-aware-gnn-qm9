import matplotlib.pyplot as plt


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

