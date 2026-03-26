# pine-fcgr-github

A cleaned and modularized version of the original single-file script for pine DNA fragment classification.

## What is included

- `train` mode for supervised model training from CSV files
- `predict` mode for FASTA-based preprocessing + inference
- FCGR, GC-content, and One-Hot feature extraction
- CNN-Transformer backbone and ablation-friendly pure CNN backbone
- Optional class weights, focal loss, SMOTE, random forest evaluation, and ensemble evaluation
- Metadata saving for reproducible GitHub release

## Project structure

```
pine-fcgr-github/
├── README.md
├── requirements.txt
├── pyproject.toml
├── scripts/
│   ├── train.py
│   └── predict.py
└── src/pine_fcgr/
    ├── __init__.py
    ├── constants.py
    ├── utils.py
    ├── features.py
    ├── datasets.py
    ├── losses.py
    ├── models.py
    ├── training.py
    ├── bioinformatics.py
    └── io_utils.py
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train from pre-split CSVs:

```bash
python scripts/train.py \
  --train-csv path/to/train.csv \
  --test-csv path/to/test.csv \
  --results-dir results \
  --model-output-path results/best_model.pth
```

Train from a single CSV and split automatically:

```bash
python scripts/train.py \
  --dataset path/to/all_data.csv \
  --test-size 0.2 \
  --results-dir results \
  --model-output-path results/best_model.pth
```

Predict from FASTA:

```bash
python scripts/predict.py \
  --input-fasta sample.fa \
  --reference-fasta ref.fa \
  --model-path results/best_model.pth \
  --output-dir predict_output
```

## Notes for GitHub release

- Do **not** commit raw sequencing data, BAM/VCF files, or large checkpoints.
- Commit only code, configs, small demo data, and documentation.
- The training script saves a sidecar JSON label map automatically so prediction can recover class names.
