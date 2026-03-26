from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from scipy import stats
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def train_step(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    num_samples = 0
    for x_fcgr, x_gc, x_onehot, y in dataloader:
        x_fcgr, x_gc, x_onehot, y = x_fcgr.to(device), x_gc.to(device), x_onehot.to(device), y.to(device)
        batch_size = x_fcgr.size(0)
        logits = model(x_fcgr, x_gc, x_onehot)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == y).sum().item()
        num_samples += batch_size
    return total_loss / max(num_samples, 1), correct / max(num_samples, 1)


def eval_step(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    num_samples = 0
    with torch.inference_mode():
        for x_fcgr, x_gc, x_onehot, y in dataloader:
            x_fcgr, x_gc, x_onehot, y = x_fcgr.to(device), x_gc.to(device), x_onehot.to(device), y.to(device)
            batch_size = x_fcgr.size(0)
            logits = model(x_fcgr, x_gc, x_onehot)
            loss = criterion(logits, y)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == y).sum().item()
            num_samples += batch_size
    return total_loss / max(num_samples, 1), correct / max(num_samples, 1)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
) -> tuple[Dict[str, List[float]], Optional[dict]]:
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_test_acc = 0.0
    best_model_state = None
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = eval_step(model, test_loader, criterion, device)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_acc)
            else:
                scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1:03d}/{epochs:03d} | lr={current_lr:.1e} | train_loss={train_loss:.4f} | "
                f"train_acc={train_acc*100:.2f}% | test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}%"
            )
    return history, best_model_state


def predict_probabilities(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    preds_list = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                x_fcgr, x_gc, x_onehot, _ = batch
            else:
                x_fcgr, x_gc, x_onehot = batch
            x_fcgr, x_gc, x_onehot = x_fcgr.to(device), x_gc.to(device), x_onehot.to(device)
            logits = model(x_fcgr, x_gc, x_onehot)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            probs_list.append(probs.cpu().numpy())
            preds_list.append(preds.cpu().numpy())
    if not probs_list:
        return np.array([]), np.array([])
    return np.concatenate(probs_list), np.concatenate(preds_list)


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device, idx_to_class: Dict[int, str], output_prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    y_true_list = []
    y_probs_list = []
    y_preds_list = []
    with torch.no_grad():
        for x_fcgr, x_gc, x_onehot, y in tqdm(dataloader, desc="evaluate", leave=False):
            x_fcgr, x_gc, x_onehot, y = x_fcgr.to(device), x_gc.to(device), x_onehot.to(device), y.to(device)
            logits = model(x_fcgr, x_gc, x_onehot)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true_list.append(y.cpu().numpy())
            y_probs_list.append(probs.cpu().numpy())
            y_preds_list.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
    y_probs = np.concatenate(y_probs_list) if y_probs_list else np.array([])
    y_preds = np.concatenate(y_preds_list) if y_preds_list else np.array([])
    acc = float(np.mean(y_true == y_preds)) if len(y_true) else 0.0

    report_path = f"{output_prefix}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f}\n")
        if len(y_true):
            present_labels = np.unique(y_true)
            present_target_names = [idx_to_class.get(label, str(label)) for label in present_labels]
            f.write("\n--- Classification Report ---\n")
            f.write(classification_report(y_true, y_preds, labels=present_labels, target_names=present_target_names, digits=4, zero_division=0))
            f.write("\n--- Confusion Matrix ---\n")
            f.write(np.array2string(confusion_matrix(y_true, y_preds, labels=present_labels)))
            if y_probs.size:
                num_classes = y_probs.shape[1]
                if num_classes > 2:
                    roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
                    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                    pr_auc = average_precision_score(y_true_bin, y_probs, average="macro")
                    f.write(f"\nROC-AUC: {roc_auc:.4f}\nPR-AUC: {pr_auc:.4f}\n")
    if MATPLOTLIB_AVAILABLE and y_probs.size and len(y_true):
        _save_curves(y_true, y_probs, idx_to_class, output_prefix)
    return y_probs, y_preds, y_true, acc


def _save_curves(y_true: np.ndarray, y_probs: np.ndarray, idx_to_class: Dict[int, str], output_prefix: str) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    num_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    class_names = [idx_to_class.get(i, f"Class {i}") for i in range(num_classes)]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    for i in range(num_classes):
        RocCurveDisplay.from_predictions(y_true_bin[:, i], y_probs[:, i], name=f"ROC ({class_names[i]})", ax=ax)
    plt.title("Multi-class ROC Curve")
    plt.grid(True)
    plt.savefig(f"{output_prefix}_roc_curve.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    for i in range(num_classes):
        PrecisionRecallDisplay.from_predictions(y_true_bin[:, i], y_probs[:, i], name=f"PR ({class_names[i]})", ax=ax)
    plt.title("Multi-class PR Curve")
    plt.grid(True)
    plt.savefig(f"{output_prefix}_pr_curve.png")
    plt.close()


def extract_features(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    all_features = []
    all_labels = []
    has_labels = None
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="extract_features", leave=False):
            if has_labels is None:
                has_labels = len(batch) == 4
            if has_labels:
                x_fcgr, x_gc, _x_onehot, y = batch
                all_labels.append(y.cpu().numpy())
            else:
                x_fcgr, x_gc, _x_onehot = batch
            x_fcgr, x_gc = x_fcgr.to(device), x_gc.to(device)
            h_fcgr = model.backbone(x_fcgr) if model.backbone is not None else torch.empty((x_fcgr.size(0), 0), device=device)
            combined = torch.cat((h_fcgr, x_gc), dim=1)
            all_features.append(combined.cpu().numpy())
    features = np.concatenate(all_features) if all_features else np.array([])
    labels = np.concatenate(all_labels) if all_labels else None
    return features, labels


def evaluate_random_forest(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, random_seed: int, use_class_weights: bool) -> dict[str, float]:
    x_train, y_train = extract_features(model, train_loader, device)
    x_test, y_test = extract_features(model, test_loader, device)
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_seed,
        n_jobs=-1,
        class_weight="balanced" if use_class_weights else None,
    )
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    return {"rf_acc": float(np.mean(pred == y_test))}


def evaluate_ensemble(all_test_preds: list[np.ndarray], all_test_probs: list[np.ndarray], y_true: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    if len(all_test_preds) <= 1:
        return result
    hard_preds_array = np.array(all_test_preds)
    final_hard_pred, _ = stats.mode(hard_preds_array, axis=0, keepdims=False)
    result["hard_vote_acc"] = float(np.mean(final_hard_pred == y_true))
    soft_probs_array = np.array(all_test_probs)
    avg_prob = np.mean(soft_probs_array, axis=0)
    final_soft_pred = np.argmax(avg_prob, axis=1)
    result["soft_vote_acc"] = float(np.mean(final_soft_pred == y_true))
    return result
