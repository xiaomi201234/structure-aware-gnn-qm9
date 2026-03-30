import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import kendalltau
import os

PROPERTY_NAMES = ["mu", "u0_atom", "homo", "lumo", "gap", "alpha", "cv"]


def normalize_property(data_list, property_idx, train_indices):
    vals = []
    for i in train_indices:
        v = data_list[i].y[:, property_idx]
        if v.numel() == 1:
            vals.append(v.item())
        else:
            vals.extend(v.tolist())
    mean, std = np.mean(vals), np.std(vals)
    if std < 1e-8:
        std = 1.0
    for i in range(len(data_list)):
        data_list[i].y[:, property_idx] = (data_list[i].y[:, property_idx] - mean) / std
    return mean, std


def train_epoch(model, loader, optimizer, criterion, device, property_idx, edge_attr=False):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        target = batch.y[:, property_idx].unsqueeze(1)
        if edge_attr and hasattr(batch, "edge_attr") and batch.edge_attr is not None:
            pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        else:
            pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, property_idx, edge_attr=False):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if edge_attr and hasattr(batch, "edge_attr") and batch.edge_attr is not None:
                out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y[:, property_idx]
            preds.append(out.cpu().numpy().flatten())
            targets.append(target.cpu().numpy().flatten())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    tau, _ = kendalltau(targets, preds)
    return {"mse": mse, "mae": mae, "r2": r2, "tau": tau}


def train_one_property(
    model,
    train_set,
    val_set,
    test_set,
    property_idx,
    save_path,
    epochs=300,
    lr=1e-3,
    batch_size=64,
    device=None,
    use_edge_attr=False,
    train_indices=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if train_indices is None:
        train_indices = list(range(len(train_set)))
    norm_mean, norm_std = normalize_property(train_set + val_set + test_set, property_idx, train_indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    best_val_r2 = -float("inf")
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device, property_idx, edge_attr=use_edge_attr)
        val_metrics = evaluate(model, val_loader, device, property_idx, edge_attr=use_edge_attr)
        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, property_idx, edge_attr=use_edge_attr)
    return {"norm_mean": norm_mean, "norm_std": norm_std, "test": test_metrics, "best_val_r2": best_val_r2}


def run_all_properties(
    model_class,
    model_kwargs,
    data_list,
    split_fn,
    save_dir,
    properties=None,
    epochs=300,
    lr=1e-3,
    batch_size=64,
    device=None,
    use_edge_attr=False,
    model_name="model",
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if properties is None:
        properties = list(range(data_list[0].y.size(1)))
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    for prop_idx in properties:
        prop_name = PROPERTY_NAMES[prop_idx] if prop_idx < len(PROPERTY_NAMES) else str(prop_idx)
        train_set, val_set, test_set = split_fn(data_list)
        train_indices = list(range(len(train_set)))
        model = model_class(**model_kwargs)
        save_path = os.path.join(save_dir, f"{model_name}_{prop_name}.pt")
        metrics = train_one_property(
            model,
            train_set,
            val_set,
            test_set,
            prop_idx,
            save_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            use_edge_attr=use_edge_attr,
            train_indices=train_indices,
        )
        results[prop_name] = metrics
    return results
