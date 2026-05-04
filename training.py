import random
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import DEFAULTS, SEED
from .dataset import collate_fn
from .evaluate import evaluate_regression, evaluate_classification


def set_seed(seed: int = SEED, deterministic: bool = False):
    # Needed by CUDA/cuBLAS for deterministic behavior on some GPU kernels.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Keep deterministic preference, but do not crash on unsupported kernels.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


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
    for idx, record in enumerate(records):
        smiles = record[0]
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


def train_epoch(model, loader, optimizer, loss_fn, device, max_grad_norm=None):
    model.train()
    total_loss = 0.0
    kept_samples = 0
    skipped_batches = 0
    use_cuda = str(device).startswith("cuda")
    for lig_batch, poc_batch, int_batch, labels in loader:
        lig_batch = lig_batch.to(device, non_blocking=use_cuda)
        poc_batch = poc_batch.to(device, non_blocking=use_cuda)
        int_batch = int_batch.to(device, non_blocking=use_cuda)
        labels = labels.to(device, non_blocking=use_cuda)

        # Skip batches with invalid numeric values in graph tensors or labels.
        has_bad_inputs = (
            not torch.isfinite(lig_batch.x).all()
            or not torch.isfinite(poc_batch.x).all()
            or (lig_batch.edge_attr.numel() > 0 and not torch.isfinite(lig_batch.edge_attr).all())
            or (poc_batch.edge_attr.numel() > 0 and not torch.isfinite(poc_batch.edge_attr).all())
            or (int_batch.x.numel() > 0 and not torch.isfinite(int_batch.x).all())
            or (int_batch.edge_attr.numel() > 0 and not torch.isfinite(int_batch.edge_attr).all())
            or (hasattr(lig_batch, "mol_desc") and not torch.isfinite(lig_batch.mol_desc).all())
            or not torch.isfinite(labels).all()
        )
        if has_bad_inputs:
            skipped_batches += 1
            continue

        optimizer.zero_grad()
        preds = model(lig_batch, poc_batch, int_batch)

        finite_mask = torch.isfinite(preds) & torch.isfinite(labels)
        if not finite_mask.any():
            skipped_batches += 1
            continue
        preds = preds[finite_mask]
        labels = labels[finite_mask]

        loss = loss_fn(preds, labels)
        if not torch.isfinite(loss):
            skipped_batches += 1
            continue

        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        kept_samples += labels.size(0)

    train_loss = total_loss / max(kept_samples, 1)
    return train_loss, kept_samples, skipped_batches


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    kept_samples = 0
    skipped_batches = 0
    y_true = []
    y_pred = []
    use_cuda = str(device).startswith("cuda")
    with torch.no_grad():
        for lig_batch, poc_batch, int_batch, labels in loader:
            lig_batch = lig_batch.to(device, non_blocking=use_cuda)
            poc_batch = poc_batch.to(device, non_blocking=use_cuda)
            int_batch = int_batch.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            has_bad_inputs = (
                not torch.isfinite(lig_batch.x).all()
                or not torch.isfinite(poc_batch.x).all()
                or (lig_batch.edge_attr.numel() > 0 and not torch.isfinite(lig_batch.edge_attr).all())
                or (poc_batch.edge_attr.numel() > 0 and not torch.isfinite(poc_batch.edge_attr).all())
                or (int_batch.x.numel() > 0 and not torch.isfinite(int_batch.x).all())
                or (int_batch.edge_attr.numel() > 0 and not torch.isfinite(int_batch.edge_attr).all())
                or (hasattr(lig_batch, "mol_desc") and not torch.isfinite(lig_batch.mol_desc).all())
                or not torch.isfinite(labels).all()
            )
            if has_bad_inputs:
                skipped_batches += 1
                continue

            preds = model(lig_batch, poc_batch, int_batch)

            finite_mask = torch.isfinite(preds) & torch.isfinite(labels)
            if not finite_mask.any():
                skipped_batches += 1
                continue
            preds = preds[finite_mask]
            labels = labels[finite_mask]

            loss = loss_fn(preds, labels)
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            total_loss += loss.item() * labels.size(0)
            kept_samples += labels.size(0)
            y_true.append(labels.detach().cpu())
            y_pred.append(preds.detach().cpu())

    if kept_samples == 0:
        return float("inf"), float("nan"), float("nan"), kept_samples, skipped_batches

    valid_loss = total_loss / kept_samples
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    return valid_loss, rmse, mae, kept_samples, skipped_batches


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
    checkpoint_path: str | None = None,
    log_path: str | None = None,
    num_workers: int = 0,
    deterministic: bool = False,
    seed: int = SEED,
    max_grad_norm: float | None = None,
    use_warmup: bool = False,
    warmup_epochs: int = 5,
    use_cosine_anneal: bool = False,
):
    set_seed(seed=seed, deterministic=deterministic)
    if str(device).startswith("cuda"):
        # Favor speed on Ampere+ GPUs while keeping numerics stable for training.
        torch.set_float32_matmul_precision("high")

    pin_memory = str(device).startswith("cuda")
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=DEFAULTS["weight_decay"])
    
    # Setup scheduler
    if use_warmup and use_cosine_anneal:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 1e-2
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    elif use_warmup:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
    elif use_cosine_anneal:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 1e-2
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    if mode == "classification":
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        loss_fn = nn.SmoothL1Loss()

    best_loss = float("inf")
    best_model = None
    epochs_no_improve = 0

    def _append_log(line: str):
        if log_path is None:
            return
        try:
            with open(log_path, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_kept_samples, train_skipped_batches = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            max_grad_norm=max_grad_norm,
        )
        valid_loss, valid_rmse, valid_mae, valid_kept_samples, valid_skipped_batches = validate(
            model,
            valid_loader,
            loss_fn,
            device,
        )
        
        # Step scheduler based on type
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_loss)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(
            (
                f"Epoch {epoch+1}/{epochs}, train_loss={train_loss:.4f}, "
                f"valid_loss={valid_loss:.4f}, valid_rmse={valid_rmse:.4f}, "
                f"valid_mae={valid_mae:.4f}, lr={current_lr:.2e}, time={elapsed:.1f}s, "
                f"train_kept={train_kept_samples}, train_skipped_batches={train_skipped_batches}, "
                f"valid_kept={valid_kept_samples}, valid_skipped_batches={valid_skipped_batches}"
            ),
            flush=True,
        )
        _append_log(
            (
                f"Epoch {epoch+1}/{epochs}, train_loss={train_loss:.4f}, "
                f"valid_loss={valid_loss:.4f}, valid_rmse={valid_rmse:.4f}, "
                f"valid_mae={valid_mae:.4f}, lr={current_lr:.2e}, time={elapsed:.1f}s, "
                f"train_kept={train_kept_samples}, train_skipped_batches={train_skipped_batches}, "
                f"valid_kept={valid_kept_samples}, valid_skipped_batches={valid_skipped_batches}"
            )
        )

        if train_kept_samples == 0:
            print("Stopping: no finite training samples remained in this epoch.", flush=True)
            _append_log("Stopping: no finite training samples remained in this epoch.")
            break

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "best_valid_loss": best_loss,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
                print(f"  ↳ saved best checkpoint: {checkpoint_path}", flush=True)
                _append_log(f"  saved_best_checkpoint: {checkpoint_path} @ epoch={epoch+1}, valid_loss={best_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs.", flush=True)
                _append_log(f"Early stopping after {epoch+1} epochs.")
                break

    if best_model is not None:
        model.load_state_dict(best_model)

    metrics = evaluate_classification(model, valid_loader, device) if mode == "classification" else evaluate_regression(model, valid_loader, device)
    print(metrics, flush=True)
    _append_log(f"Final metrics: {metrics}")
    return model


if __name__ == "__main__":
    print("This training module implements the PDF model training procedure.")
    print("Prepare a LigandPocketDataset and call train(model, train_dataset, valid_dataset, mode='regression' or 'classification').")
