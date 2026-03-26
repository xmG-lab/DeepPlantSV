from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class FeatureBundle:
    fcgr: np.ndarray
    gc: np.ndarray
    onehot: np.ndarray
    labels: Optional[np.ndarray] = None


class SeqDataset(Dataset):
    """Dataset for FCGR, GC-content, and One-Hot features."""

    def __init__(
        self,
        fcgr_features: np.ndarray,
        gc_features: np.ndarray,
        onehot_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        if not (fcgr_features.shape[0] == gc_features.shape[0] == onehot_features.shape[0]):
            raise ValueError(
                f"Feature count mismatch: FCGR={fcgr_features.shape[0]}, GC={gc_features.shape[0]}, OneHot={onehot_features.shape[0]}"
            )
        if labels is not None and fcgr_features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Feature count {fcgr_features.shape[0]} does not match label count {labels.shape[0]}"
            )
        self.fcgr_features = fcgr_features
        self.gc_features = gc_features
        self.onehot_features = onehot_features
        self.labels = labels
        self.has_labels = labels is not None

    def __len__(self) -> int:
        return self.fcgr_features.shape[0]

    def __getitem__(self, index: int) -> Union[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        fcgr_feat = torch.from_numpy(self.fcgr_features[index]).unsqueeze(0).float()
        gc_feat = torch.tensor([self.gc_features[index]], dtype=torch.float32)
        onehot_feat = torch.from_numpy(self.onehot_features[index]).float()
        if self.has_labels:
            return fcgr_feat, gc_feat, onehot_feat, int(self.labels[index])
        return fcgr_feat, gc_feat, onehot_feat
