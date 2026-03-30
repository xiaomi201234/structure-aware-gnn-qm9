import torch
import numpy as np


def random_split(dataset, train_ratio=0.8, val_ratio=0.1, seed=None):
    N = len(dataset)
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    perm = torch.randperm(N, generator=gen)
    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))
    train_set = [dataset[i] for i in perm[:train_end]]
    val_set = [dataset[i] for i in perm[train_end:val_end]]
    test_set = [dataset[i] for i in perm[val_end:]]
    return train_set, val_set, test_set


def scaffold_split(dataset, idx_path):
    idx_dict = torch.load(idx_path, weights_only=False)
    train_set = [dataset[i] for i in idx_dict["train"]]
    val_set = [dataset[i] for i in idx_dict["val"]]
    test_set = [dataset[i] for i in idx_dict["test"]]
    return train_set, val_set, test_set


def scaffold_split_from_df(dataset, df, scaffold_col="scaffold", train_ratio=0.8, val_ratio=0.1, seed=42):
    groups = df.groupby(scaffold_col).indices
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_idx, val_idx, test_idx = [], [], []
    for sc in keys:
        idx = groups[sc]
        if len(train_idx) + len(idx) <= n_train:
            train_idx.extend(idx)
        elif len(val_idx) + len(idx) <= n_val:
            val_idx.extend(idx)
        else:
            test_idx.extend(idx)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]
    return train_set, val_set, test_set
