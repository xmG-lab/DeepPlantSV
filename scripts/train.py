from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from pine_fcgr.datasets import SeqDataset
from pine_fcgr.io_utils import apply_smote_triplet, build_feature_triplet, prepare_label_splits
from pine_fcgr.losses import FocalLoss
from pine_fcgr.models import ModelOptions, MultiBranchNetwork
from pine_fcgr.training import evaluate_ensemble, evaluate_model, evaluate_random_forest, fit_model
from pine_fcgr.utils import ensure_dir, infer_config_path, infer_label_map_path, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train pine DNA classification models.")
    parser.add_argument("--dataset", type=str, default=None, help="Single CSV with sequence and label columns.")
    parser.add_argument("--train-csv", type=str, default=None, help="Pre-split training CSV.")
    parser.add_argument("--test-csv", type=str, default=None, help="Pre-split testing CSV.")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--apply-smote", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--number-of-models", type=int, default=1)
    parser.add_argument("--evaluate-random-forest", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model-output-path", type=str, default="results/best_model.pth")
    parser.add_argument("--no-fcgr", action="store_true")
    parser.add_argument("--no-onehot", action="store_true")
    parser.add_argument("--no-gc", action="store_true")
    parser.add_argument("--ablate-transformer", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.random_seed)
    results_dir = ensure_dir(args.results_dir)

    prepared = prepare_label_splits(args.dataset, args.train_csv, args.test_csv, args.test_size, args.random_seed)
    print(f"classes: {prepared.class_to_idx}")
    print(f"train label distribution: {Counter(prepared.y_train)}")
    print(f"test label distribution: {Counter(prepared.y_test)}")

    x_fcgr_train, x_gc_train, x_onehot_train = build_feature_triplet(prepared.train_df, args.k)
    x_fcgr_test, x_gc_test, x_onehot_test = build_feature_triplet(prepared.test_df, args.k)

    if args.apply_smote:
        x_fcgr_train, x_gc_train, x_onehot_train, y_train = apply_smote_triplet(
            x_fcgr_train, x_gc_train, x_onehot_train, prepared.y_train, args.random_seed
        )
    else:
        y_train = prepared.y_train
    y_test = prepared.y_test

    criterion = None
    if args.use_focal_loss:
        criterion = FocalLoss(gamma=2, alpha=0.25).to(device)
    elif args.use_class_weights:
        unique_labels = np.unique(y_train)
        if len(unique_labels) > 1:
            class_weights_values = compute_class_weight("balanced", classes=unique_labels, y=y_train)
            weights = np.ones(prepared.num_classes)
            for label_idx, weight_val in zip(unique_labels, class_weights_values):
                weights[label_idx] = weight_val
            criterion = CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    if criterion is None:
        criterion = CrossEntropyLoss()

    train_dataset = SeqDataset(x_fcgr_train, x_gc_train, x_onehot_train, y_train)
    test_dataset = SeqDataset(x_fcgr_test, x_gc_test, x_onehot_test, y_test)
    actual_num_workers = min(args.num_workers, max(1, (Path('/proc/cpuinfo').exists() and 2) or 1))
    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_test_preds = []
    all_test_probs = []
    best_overall_test_acc = 0.0
    best_overall_model_state = None
    last_model = None
    all_metrics = []

    options = ModelOptions(
        use_fcgr=not args.no_fcgr,
        use_gc=not args.no_gc,
        use_onehot=not args.no_onehot,
        ablate_transformer=args.ablate_transformer,
    )

    for i in range(args.number_of_models):
        current_seed = args.random_seed + i
        set_seed(current_seed)
        model = MultiBranchNetwork(
            k_value=args.k,
            backbone_embedding_dim=args.embedding_dim,
            gc_dim=1,
            class_num=prepared.num_classes,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            transformer_ff_dim=args.transformer_ff_dim,
            options=options,
        ).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10)

        history, best_state = fit_model(model, train_loader, test_loader, optimizer, scheduler, criterion, args.num_epochs, device)
        if best_state is not None:
            model.load_state_dict(best_state)
        output_prefix = str(results_dir / f"{timestamp}_model_{i+1}")
        test_probs, test_preds, test_true, test_acc = evaluate_model(model, test_loader, device, prepared.idx_to_class, output_prefix)
        all_test_preds.append(test_preds)
        all_test_probs.append(test_probs)
        last_model = model
        metrics = {"model_index": i + 1, "test_acc": test_acc, **{k: v[-1] for k, v in history.items() if v}}
        all_metrics.append(metrics)
        if test_acc > best_overall_test_acc:
            best_overall_test_acc = test_acc
            best_overall_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ensemble_metrics = evaluate_ensemble(all_test_preds, all_test_probs, test_true) if len(all_test_preds) > 1 else {}
    rf_metrics = evaluate_random_forest(last_model, train_loader, test_loader, device, args.random_seed, args.use_class_weights) if args.evaluate_random_forest and last_model is not None else {}

    model_output_path = Path(args.model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    if best_overall_model_state is not None:
        torch.save(best_overall_model_state, model_output_path)
    label_map_path = infer_label_map_path(model_output_path)
    config_path = infer_config_path(model_output_path)
    save_json(prepared.idx_to_class, label_map_path)
    save_json(vars(args), config_path)
    save_json({
        "timestamp": timestamp,
        "best_overall_test_acc": best_overall_test_acc,
        "per_model_metrics": all_metrics,
        "ensemble_metrics": ensemble_metrics,
        "rf_metrics": rf_metrics,
    }, results_dir / f"{timestamp}_summary.json")

    print(f"best model saved to: {model_output_path}")
    print(f"label map saved to: {label_map_path}")
    print(f"config saved to: {config_path}")


if __name__ == "__main__":
    main()
