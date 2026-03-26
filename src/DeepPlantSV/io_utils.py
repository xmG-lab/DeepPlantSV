from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from .constants import MAX_LENGTH
from .features import generate_fcgr_features, generate_gc_features, generate_onehot_features

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = None
    IMBLEARN_AVAILABLE = False


REQUIRED_COLUMNS = {"sequence", "label"}


@dataclass
class PreparedData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    num_classes: int


def read_csv_checked(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV file {path} is missing required columns: {sorted(missing)}")
    return df


def prepare_label_splits(dataset: Optional[str], train_csv: Optional[str], test_csv: Optional[str], test_size: float, random_seed: int) -> PreparedData:
    if train_csv and test_csv:
        train_df = read_csv_checked(train_csv).copy()
        test_df = read_csv_checked(test_csv).copy()
        records_df_combined = pd.concat([train_df, test_df], ignore_index=True)
    elif dataset:
        full_df = read_csv_checked(dataset).copy()
        train_df, test_df = train_test_split(
            full_df,
            test_size=test_size,
            random_state=random_seed,
            stratify=full_df["label"].astype(str),
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        records_df_combined = full_df
    else:
        raise ValueError("Either dataset or (train_csv and test_csv) must be provided.")

    records_df_combined["label"] = records_df_combined["label"].astype(str)
    class_names = sorted(records_df_combined["label"].unique())
    class_to_idx = {label: i for i, label in enumerate(class_names)}
    idx_to_class = {i: label for label, i in class_to_idx.items()}

    train_df["label"] = train_df["label"].astype(str)
    test_df["label"] = test_df["label"].astype(str)
    y_train = train_df["label"].map(class_to_idx).to_numpy()
    y_test = test_df["label"].map(class_to_idx).to_numpy()

    return PreparedData(
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        y_test=y_test,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        num_classes=len(class_names),
    )


def build_feature_triplet(df: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        generate_fcgr_features(df, k),
        generate_gc_features(df),
        generate_onehot_features(df, MAX_LENGTH),
    )


def apply_smote_triplet(
    fcgr: np.ndarray, gc: np.ndarray, onehot: np.ndarray, labels: np.ndarray, random_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn is not installed, but --apply-smote was requested.")

    n = labels.shape[0]
    fcgr_shape = fcgr.shape[1:]
    onehot_shape = onehot.shape[1:]
    flat = np.concatenate([fcgr.reshape(n, -1), gc.reshape(n, -1), onehot.reshape(n, -1)], axis=1)
    smote = SMOTE(random_state=random_seed)
    flat_res, y_res = smote.fit_resample(flat, labels)

    fcgr_size = int(np.prod(fcgr_shape))
    gc_size = gc.shape[1] if gc.ndim > 1 else 1
    onehot_size = int(np.prod(onehot_shape))

    fcgr_res = flat_res[:, :fcgr_size].reshape(-1, *fcgr_shape).astype(np.float32)
    gc_res = flat_res[:, fcgr_size : fcgr_size + gc_size].reshape(-1).astype(np.float32)
    onehot_res = flat_res[:, fcgr_size + gc_size : fcgr_size + gc_size + onehot_size].reshape(-1, *onehot_shape).astype(np.float32)
    return fcgr_res, gc_res, onehot_res, y_res.astype(labels.dtype)
