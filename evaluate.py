import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
import torch


def evaluate_regression(model, loader, device="cpu"):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for lig_batch, poc_batch, int_batch, labels in loader:
            lig_batch = lig_batch.to(device)
            poc_batch = poc_batch.to(device)
            int_batch = int_batch.to(device)
            labels = labels.to(device)
            preds = model(lig_batch, poc_batch, int_batch)
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "y_true": y_true, "y_pred": y_pred}


def evaluate_classification(model, loader, device="cpu"):
    model.eval()
    y_true = []
    y_scores = []
    with torch.no_grad():
        for lig_batch, poc_batch, int_batch, labels in loader:
            lig_batch = lig_batch.to(device)
            poc_batch = poc_batch.to(device)
            int_batch = int_batch.to(device)
            labels = labels.to(device)
            logits = model(lig_batch, poc_batch, int_batch)
            scores = torch.sigmoid(logits)
            y_true.append(labels.cpu().numpy())
            y_scores.append(scores.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)
    roc = roc_auc_score(y_true.astype(int), y_scores)
    pr = average_precision_score(y_true.astype(int), y_scores)
    return {"roc_auc": roc, "pr_auc": pr, "y_true": y_true, "y_scores": y_scores}


def enrichment_factor(y_true, y_scores, top_fraction: float = 0.01):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)
    if n == 0:
        return 0.0
    top_k = max(1, int(np.ceil(n * top_fraction)))
    order = np.argsort(-y_scores)
    top_labels = y_true[order][:top_k]
    ef = top_labels.sum() / (top_k * (y_true.sum() / n) + 1e-9)
    return ef


def bedroc_score(y_true, y_scores, alpha: float = 20.0):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)
    a = np.sum(y_true)
    if n == 0 or a == 0:
        return 0.0
    order = np.argsort(-y_scores)
    ranks = np.arange(1, n + 1)
    active_ranks = ranks[y_true[order] == 1]
    if len(active_ranks) == 0:
        return 0.0
    ra = a / n
    sum_exp = np.sum(np.exp(-alpha * active_ranks / n))
    coefficient = (alpha / (1.0 - np.exp(-alpha))) * (1.0 / (n * ra))
    return sum_exp * coefficient


if __name__ == "__main__":
    print("Evaluation helpers for regression, classification, EF and BEDROC.")
