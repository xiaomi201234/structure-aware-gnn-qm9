import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_DEFAULT_COLORS = ["#b6d7a8", "#d9d2e9", "#ffe599", "#cfe2f3", "#d9d9d9", "#ead1dc", "#c9a0a0", "#7d9fb5"]


def plot_sdi_bar(pivot_sdi, colors=None, figsize=(10, 6)):
    if colors is None:
        colors = _DEFAULT_COLORS
    fig, ax = plt.subplots(figsize=figsize)
    pivot_sdi.plot(kind="bar", ax=ax, color=colors[:pivot_sdi.shape[1]], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("SDI")
    ax.set_title("Structural Degradation Index across Properties")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig, ax


def plot_sdi_ci(boot_df, by="property", property_order=None, model_colors=None, figsize=(8, 5)):
    if by == "both":
        if model_colors is None:
            model_colors = _DEFAULT_COLORS
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if property_order is not None:
            boot_df[0] = boot_df[0].reindex(property_order)
        x = np.arange(len(boot_df[0]))
        axes[0].vlines(x, boot_df[0]["lower"], boot_df[0]["upper"], color="#8fbcd4", linewidth=2)
        axes[0].scatter(x, boot_df[0]["mean"], color="#4E79A7", s=70, zorder=3)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(boot_df[0].index, rotation=45)
        axes[0].set_ylabel("Average SDI")
        axes[0].set_ylim(0.04, 0.14)
        axes[1].bar(boot_df[1]["Model"], boot_df[1]["mean"],
                    yerr=[boot_df[1]["err_low"], boot_df[1]["err_high"]], capsize=3,
                    color=model_colors[:len(boot_df[1])], edgecolor="black", linewidth=0.6)
        axes[1].set_ylabel("Average SDI")
        axes[1].set_ylim(0, 0.13)
        axes[1].tick_params(axis="x", rotation=30)
        plt.tight_layout()
        return fig, axes
    if property_order is not None and by == "property":
        boot_df = boot_df.reindex(property_order)
    fig, ax = plt.subplots(figsize=figsize)
    if by == "property":
        x = np.arange(len(boot_df))
        ax.vlines(x, boot_df["lower"], boot_df["upper"], color="#8fbcd4", linewidth=2)
        ax.scatter(x, boot_df["mean"], color="#4E79A7", s=70, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(boot_df.index, rotation=45)
        ax.set_ylim(0.04, 0.14)
    else:
        colors = model_colors or _DEFAULT_COLORS
        ax.bar(boot_df["Model"], boot_df["mean"],
               yerr=[boot_df["err_low"], boot_df["err_high"]], capsize=3,
               color=colors[:len(boot_df)], edgecolor="black", linewidth=0.6)
        ax.set_ylim(0, 0.13)
        ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Average SDI")
    plt.tight_layout()
    return fig, ax


def plot_sdi_heatmap(pivot_sdi, figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot_sdi, annot=True, fmt=".3f", cmap="Blues", ax=ax)
    ax.set_title("SDI Heatmap")
    plt.tight_layout()
    return fig, ax


def plot_property_bootstrap_bars(boot_df, property_order=None, figsize=(8, 5)):
    if property_order is not None:
        boot_df = boot_df.reindex(property_order)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(boot_df.index, boot_df["mean"], yerr=[boot_df["err_low"], boot_df["err_high"]], capsize=5, color="#93c47d")
    ax.set_ylabel("Average SDI")
    ax.set_title("Property Sensitivity with 95% CI")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, ax


def plot_rsi_model_property(rsi_model, rsi_property, model_order=None, property_order=None, colors=None):
    if colors is None:
        colors = ["#c9a0a0", "#d9d9d9", "#ead1dc", "#7d9fb5", "#ffe599", "#cfe2f3", "#d9d2e9", "#b6d7a8"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rsi_m = rsi_model.reindex(model_order) if model_order else rsi_model
    axes[0].bar(rsi_m.index, rsi_m.values, color=colors[:len(rsi_m)], edgecolor="black", linewidth=0.6)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Average RSI")
    axes[0].set_ylim(0.8, 1.05)
    rsi_p = rsi_property.reindex(property_order) if property_order else rsi_property
    x = np.arange(len(rsi_p))
    axes[1].scatter(x, rsi_p.values, color="#d8a7a7", s=60)
    axes[1].plot(x, rsi_p.values, color="#d8a7a7", linewidth=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(rsi_p.index, rotation=45)
    axes[1].set_ylabel("Average RSI")
    plt.tight_layout()
    return fig, axes


def plot_ssr_scatter(properties, models, ssr_matrix, colors=None, figsize=(7, 4)):
    if colors is None:
        colors = ["#d8a7a7", "#9bc48a", "#7ea6c2", "#f1cf6a", "#b8a6d9"]
    x = np.arange(len(properties))
    offset = np.linspace(-0.25, 0.25, len(models))
    fig, ax = plt.subplots(figsize=figsize)
    for i, model in enumerate(models):
        ax.scatter(x + offset[i], ssr_matrix[:, i], s=60, color=colors[i % len(colors)], label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(properties)
    ax.set_ylabel("SSR")
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False)
    plt.tight_layout()
    return fig, ax


def plot_embedding_pca(z, scaffold_ids, figsize=(5, 4), title="Embedding PCA"):
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(z[:, 0], z[:, 1], c=scaffold_ids, cmap="tab20", s=8, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label="Scaffold")
    plt.tight_layout()
    return fig, ax


def plot_structure_sensitivity_curve(sens_series, figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sens_series.index, sens_series.values, marker="o")
    ax.set_ylabel("Structure Sensitivity (MAE_scaffold / MAE_random)")
    ax.set_xlabel("Property")
    plt.tight_layout()
    return fig, ax


def plot_rare_scaffold_bar(data, x="scaffold_bucket", y="error", hue="model", figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax)
    ax.set_xlabel("Scaffold Frequency Category")
    ax.set_ylabel("Prediction Error")
    ax.set_title("Rare Scaffold Failure Analysis")
    plt.tight_layout()
    return fig, ax


def plot_rank_curves(rank_matrix, model_names, x_labels=None, colors=None, figsize=(6, 4)):
    if colors is None:
        colors = ["#3f6b86", "#c06c84", "#5fa8d3", "#e3b505", "#8a6fb3", "#5a9e4b"]
    n_bins = rank_matrix.shape[1]
    x = np.arange(len(x_labels)) if x_labels is not None else np.arange(1, n_bins + 1)
    fig, ax = plt.subplots(figsize=figsize)
    for i, name in enumerate(model_names):
        ax.plot(x, rank_matrix[i], color=colors[i % len(colors)], linewidth=3, label=name)
    ax.invert_yaxis()
    ax.set_xlabel("Scaffold Frequency")
    ax.set_ylabel("Model Rank")
    if x_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    plt.tight_layout()
    return fig, ax


def plot_rank_heatmap(rank_df, figsize=(10, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(rank_df, cmap="rocket_r", ax=ax)
    ax.set_xlabel("Scaffold Frequency")
    ax.set_ylabel("Model")
    plt.tight_layout()
    return fig, ax


def plot_sdi_rsi_four_panels(boot_sdi_prop, boot_sdi_model, rsi_model, rsi_property, property_order=None, model_order=None, model_colors=None):
    if model_colors is None:
        model_colors = _DEFAULT_COLORS
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    if property_order is not None:
        boot_sdi_prop = boot_sdi_prop.reindex(property_order)
    x = np.arange(len(boot_sdi_prop))
    axes[0, 0].vlines(x, boot_sdi_prop["lower"], boot_sdi_prop["upper"], color="#8fbcd4", linewidth=2)
    axes[0, 0].scatter(x, boot_sdi_prop["mean"], color="#4E79A7", s=70, zorder=3)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(boot_sdi_prop.index, rotation=45)
    axes[0, 0].set_ylabel("Average SDI")
    axes[0, 0].set_ylim(0.04, 0.14)
    axes[0, 0].set_title("Property Sensitivity (SDI)")
    bm = boot_sdi_model
    if model_order is not None:
        bm = bm.set_index("Model").reindex(model_order).dropna(subset=["mean"]).reset_index()
    axes[0, 1].bar(bm["Model"], bm["mean"], yerr=[bm["err_low"], bm["err_high"]], capsize=3, color=model_colors[:len(bm)], edgecolor="black", linewidth=0.6)
    axes[0, 1].set_ylabel("Average SDI")
    axes[0, 1].set_ylim(0, 0.13)
    axes[0, 1].tick_params(axis="x", rotation=30)
    rsi_m = rsi_model.reindex(model_order) if model_order else rsi_model
    axes[1, 0].bar(rsi_m.index, rsi_m.values, color=model_colors[:len(rsi_m)], edgecolor="black", linewidth=0.6)
    axes[1, 0].set_ylabel("Average RSI")
    axes[1, 0].set_ylim(0.8, 1.05)
    axes[1, 0].tick_params(axis="x", rotation=30)
    rsi_p = rsi_property.reindex(property_order) if property_order else rsi_property
    xp = np.arange(len(rsi_p))
    axes[1, 1].scatter(xp, rsi_p.values, color="#d8a7a7", s=60)
    axes[1, 1].plot(xp, rsi_p.values, color="#d8a7a7", linewidth=2)
    axes[1, 1].set_xticks(xp)
    axes[1, 1].set_xticklabels(rsi_p.index, rotation=45)
    axes[1, 1].set_ylabel("Average RSI")
    plt.tight_layout()
    return fig, axes


def plot_sdi_rsi_two_panels_model(boot_sdi_model, rsi_model, model_order=None, model_colors=None):
    if model_colors is None:
        model_colors = _DEFAULT_COLORS
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bm = boot_sdi_model
    if model_order is not None:
        bm = bm.set_index("Model").reindex(model_order).dropna(subset=["mean"]).reset_index()
    axes[0].bar(bm["Model"], bm["mean"], yerr=[bm["err_low"], bm["err_high"]], capsize=3, color=model_colors[:len(bm)], edgecolor="black", linewidth=0.6)
    axes[0].set_ylabel("Average SDI")
    axes[0].set_ylim(0, 0.13)
    axes[0].tick_params(axis="x", rotation=30)
    rsi_m = rsi_model.reindex(model_order) if model_order else rsi_model
    axes[1].bar(rsi_m.index, rsi_m.values, color=model_colors[:len(rsi_m)], edgecolor="black", linewidth=0.6)
    axes[1].set_ylabel("Average RSI")
    axes[1].set_ylim(0.8, 1.05)
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig, axes


def plot_sdi_rsi_two_panels_property(boot_sdi_prop, rsi_property, property_order=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if property_order is not None:
        boot_sdi_prop = boot_sdi_prop.reindex(property_order)
    x = np.arange(len(boot_sdi_prop))
    axes[0].vlines(x, boot_sdi_prop["lower"], boot_sdi_prop["upper"], color="#8fbcd4", linewidth=2)
    axes[0].scatter(x, boot_sdi_prop["mean"], color="#4E79A7", s=70, zorder=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(boot_sdi_prop.index, rotation=45)
    axes[0].set_ylabel("Average SDI")
    axes[0].set_ylim(0.04, 0.14)
    axes[0].set_title("Property Sensitivity (SDI)")
    rsi_p = rsi_property.reindex(property_order) if property_order else rsi_property
    xp = np.arange(len(rsi_p))
    axes[1].scatter(xp, rsi_p.values, color="#d8a7a7", s=60)
    axes[1].plot(xp, rsi_p.values, color="#d8a7a7", linewidth=2)
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(rsi_p.index, rotation=45)
    axes[1].set_ylabel("Average RSI")
    plt.tight_layout()
    return fig, axes


def plot_curves_by_atoms_rings_scaffold(atoms_x, data_atoms, rings_x, data_rings, scaffold_x, data_scaffold, model_names, colors=None):
    if colors is None:
        colors = ["#d5a6bd", "#4f81bd", "#ffd966", "#9fc5e8", "#b4a7d6", "#93c47d"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for y, c, m in zip(data_atoms, colors, model_names):
        axes[0].plot(atoms_x, np.array(y), color=c, label=m)
    axes[0].set_xlabel("Number of atoms")
    axes[0].set_ylabel("Error")
    for y, c, m in zip(data_rings, colors, model_names):
        axes[1].plot(rings_x, np.array(y), color=c, label=m)
    axes[1].set_xlabel("Ring count")
    axes[1].set_ylabel("Error")
    for y, c, m in zip(data_scaffold, colors, model_names):
        axes[2].plot(scaffold_x, np.array(y), color=c, label=m)
    axes[2].set_xlabel("Scaffold size")
    axes[2].set_ylabel("Error")
    for ax in axes:
        ax.legend()
        ax.spines["top"].set_visible(False)
    plt.tight_layout()
    return fig, axes
