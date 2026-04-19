import random
import os

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from .config import DEFAULTS, SEED
    from .dataset import collate_fn
    from .evaluate import evaluate_regression, evaluate_classification
except ImportError:
    from config import DEFAULTS, SEED
    from dataset import collate_fn
    from evaluate import evaluate_regression, evaluate_classification


def set_seed(seed: int = SEED):
    # Needed by CUDA/cuBLAS for deterministic behavior on some GPU kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep deterministic preference, but do not crash on unsupported kernels.
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


def scaffold_split(records, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    scaffolds = {}
    for idx, (smiles, _, _) in enumerate(records):
        scaffold = build_scaffold(smiles)
        scaffolds.setdefault(scaffold, []).append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    train_ids, valid_ids, test_ids = [], [], []
    n_train = int(len(records) * frac_train)
    n_valid = int(len(records) * frac_valid)
    for indices in scaffold_sets:
        if len(train_ids) + len(indices) <= n_train:
            train_ids.extend(indices)
        elif len(valid_ids) + len(indices) <= n_valid:
            valid_ids.extend(indices)
        else:
            test_ids.extend(indices)

    if len(test_ids) == 0 and len(valid_ids) > 0:
        test_ids = valid_ids[-len(indices):]
        valid_ids = valid_ids[:-len(indices)]

    return [records[i] for i in train_ids], [records[i] for i in valid_ids], [records[i] for i in test_ids]


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for lig_batch, poc_batch, int_batch, labels in loader:
        lig_batch = lig_batch.to(device)
        poc_batch = poc_batch.to(device)
        int_batch = int_batch.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(lig_batch, poc_batch, int_batch)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for lig_batch, poc_batch, int_batch, labels in loader:
            lig_batch = lig_batch.to(device)
            poc_batch = poc_batch.to(device)
            int_batch = int_batch.to(device)
            labels = labels.to(device)
            preds = model(lig_batch, poc_batch, int_batch)
            loss = loss_fn(preds, labels)
            total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


def train(
    model,
    train_dataset,
    valid_dataset,
    epochs: int = DEFAULTS["epochs"],
    lr: float = DEFAULTS["lr"],
    batch_size: int = DEFAULTS["batch_size"],
    device: str = "cpu",
    mode: str = "regression",
    pos_weight: float = 1.0,
    patience: int = 5,
):
    set_seed()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=DEFAULTS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    if mode == "classification":
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_model = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss = validate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)
        print(f"Epoch {epoch+1}/{epochs}, train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}", flush=True)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs.", flush=True)
                break

    if best_model is not None:
        model.load_state_dict(best_model)

    metrics = evaluate_classification(model, valid_loader, device) if mode == "classification" else evaluate_regression(model, valid_loader, device)
    print(metrics, flush=True)
    return model


if __name__ == "__main__":
    print("This training module implements the PDF model training procedure.")
    print("Prepare a LigandPocketDataset and call train(model, train_dataset, valid_dataset, mode='regression' or 'classification').")
