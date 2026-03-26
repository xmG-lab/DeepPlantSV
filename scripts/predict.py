from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pine_fcgr.bioinformatics import check_tool_exists, preprocess_for_prediction
from pine_fcgr.constants import DEFAULT_BCFTOOLS_PATH, DEFAULT_BWA_PATH, DEFAULT_SAMTOOLS_PATH
from pine_fcgr.datasets import SeqDataset
from pine_fcgr.features import generate_fcgr_features, generate_gc_features, generate_onehot_features
from pine_fcgr.io_utils import build_feature_triplet
from pine_fcgr.models import ModelOptions, MultiBranchNetwork
from pine_fcgr.training import predict_probabilities
from pine_fcgr.utils import infer_config_path, infer_label_map_path, load_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict pine DNA fragment classes from FASTA.")
    parser.add_argument("--input-fasta", required=True, type=str)
    parser.add_argument("--reference-fasta", required=True, type=str)
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--label-map", default=None, type=str)
    parser.add_argument("--config-path", default=None, type=str)
    parser.add_argument("--index-dir", default=None, type=str)
    parser.add_argument("--padding", type=int, default=1000)
    parser.add_argument("--output-dir", default="predict_output", type=str)
    parser.add_argument("--output-predictions", default="predictions.csv", type=str)
    parser.add_argument("--bwa-path", default=DEFAULT_BWA_PATH, type=str)
    parser.add_argument("--samtools-path", default=DEFAULT_SAMTOOLS_PATH, type=str)
    parser.add_argument("--bcftools-path", default=DEFAULT_BCFTOOLS_PATH, type=str)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path, name in [(args.bwa_path, "bwa"), (args.samtools_path, "samtools"), (args.bcftools_path, "bcftools")]:
        if not check_tool_exists(path, name):
            raise SystemExit(f"Required tool is not available: {name}")

    predict_df = preprocess_for_prediction(
        input_fasta=args.input_fasta,
        reference_fasta=args.reference_fasta,
        padding=args.padding,
        output_dir=args.output_dir,
        bwa_path=args.bwa_path,
        samtools_path=args.samtools_path,
        bcftools_path=args.bcftools_path,
        index_dir=args.index_dir,
    )
    if predict_df is None or predict_df.empty:
        raise SystemExit("No valid sequences were produced for prediction.")

    model_path = Path(args.model_path)
    label_map_path = Path(args.label_map) if args.label_map else infer_label_map_path(model_path)
    config_path = Path(args.config_path) if args.config_path else infer_config_path(model_path)
    idx_to_class = load_json(label_map_path)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    config = load_json(config_path)

    options = ModelOptions(
        use_fcgr=not config.get("no_fcgr", False),
        use_gc=not config.get("no_gc", False),
        use_onehot=not config.get("no_onehot", False),
        ablate_transformer=config.get("ablate_transformer", False),
    )

    model = MultiBranchNetwork(
        k_value=config.get("k", 6),
        backbone_embedding_dim=config.get("embedding_dim", 512),
        gc_dim=1,
        class_num=len(idx_to_class),
        transformer_heads=config.get("transformer_heads", 8),
        transformer_layers=config.get("transformer_layers", 2),
        transformer_ff_dim=config.get("transformer_ff_dim", 512),
        options=options,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x_fcgr_pred = generate_fcgr_features(predict_df, config.get("k", 6))
    x_gc_pred = generate_gc_features(predict_df)
    x_onehot_pred = generate_onehot_features(predict_df, config.get("max_length", 2100))

    predict_dataset = SeqDataset(x_fcgr_pred, x_gc_pred, x_onehot_pred, labels=None)
    predict_loader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    pred_probs, pred_labels_idx = predict_probabilities(model, predict_loader, device)

    predict_df["predicted_label_idx"] = pred_labels_idx
    predict_df["predicted_label_name"] = [idx_to_class.get(int(idx), f"unknown_{idx}") for idx in pred_labels_idx]
    for i in range(len(idx_to_class)):
        class_name = idx_to_class.get(i, f"class_{i}")
        predict_df[f"probability_{class_name}"] = pred_probs[:, i]

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv_path = Path(args.output_dir) / args.output_predictions
    predict_df.to_csv(output_csv_path, index=False)
    print(f"prediction saved to: {output_csv_path}")


if __name__ == "__main__":
    main()
