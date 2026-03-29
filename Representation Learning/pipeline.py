import torch
from torch_geometric.loader import DataLoader

from .scaffold_utils import scaffold_train_test_split, add_scaffold_columns
from .embeddings import compute_ssr


def extract_embeddings(model, data_list, indices, batch_size=64, device=None):
    if device is None:
        device = next(model.parameters()).device
    subset = [data_list[i] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.get_embedding(batch.x, batch.edge_index, batch.batch)
            embeddings.append(z.cpu())
    return torch.cat(embeddings, dim=0).numpy()


def run_ssr_pipeline(model, data_list, df, smiles_col="smiles", train_frac=0.8, seed=42, batch_size=64, device=None):
    df = add_scaffold_columns(df, smiles_col=smiles_col)
    train_idx, test_idx = scaffold_train_test_split(df, scaffold_col="scaffold", train_frac=train_frac, seed=seed)
    embeddings = extract_embeddings(model, data_list, test_idx, batch_size=batch_size, device=device)
    scaffolds = df.iloc[test_idx]["scaffold"].values
    ssr = compute_ssr(embeddings, scaffolds)
    return ssr, embeddings, scaffolds, train_idx, test_idx
