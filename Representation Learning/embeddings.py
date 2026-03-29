import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA


def compute_ssr(embeddings, scaffolds):
    D = pairwise_distances(embeddings)
    intra, inter = [], []
    n = len(scaffolds)
    for i in range(n):
        for j in range(i + 1, n):
            if scaffolds[i] == scaffolds[j]:
                intra.append(D[i, j])
            else:
                inter.append(D[i, j])
    if not intra:
        return np.nan
    return np.mean(inter) / np.mean(intra)


def embedding_pca(embeddings, n_components=2):
    return PCA(n_components=n_components).fit_transform(embeddings)


def generate_synthetic_embedding(
    n_samples, dim, scaffold_count,
    mask_ratio=0.2, t_left=-6, t_right=4.0, spread=0.1, left_shift=0.3, curve_noise=-0.5, seed=None
):
    rng = np.random.default_rng(seed)
    t = rng.uniform(t_left, t_right, n_samples)
    x = t + rng.normal(0, 0.03, n_samples)
    y = 0.5 * np.abs(t) ** 1.35 - 2 + curve_noise * np.sin(t) + rng.normal(0, spread, n_samples)
    x -= left_shift
    emb = rng.normal(0, 0.6, (n_samples, dim))
    emb[:, 0], emb[:, 1] = x, y
    base = ((t + 4.5) / 8 * scaffold_count).astype(int)
    scaffolds = np.clip(base + rng.integers(-4, 5, n_samples), 0, scaffold_count - 1)
    mask = rng.random(n_samples) < mask_ratio
    scaffolds[mask] = rng.integers(0, scaffold_count, mask.sum())
    return PCA(n_components=2).fit_transform(emb), scaffolds
