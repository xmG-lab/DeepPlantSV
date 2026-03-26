from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer


@dataclass
class ModelOptions:
    use_fcgr: bool = True
    use_gc: bool = True
    use_onehot: bool = True
    ablate_transformer: bool = False


class BackboneModel(nn.Module):
    """CNN backbone followed by a Transformer encoder for FCGR features."""

    def __init__(
        self,
        k: int,
        embedding_dim: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 512,
    ) -> None:
        super().__init__()
        input_shape = 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        transformer_input_dim = 128
        if transformer_input_dim % transformer_heads != 0:
            raise ValueError(f"transformer_input_dim ({transformer_input_dim}) must be divisible by heads ({transformer_heads})")
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_input_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=transformer_layers)
        self.layer_norm = LayerNorm(transformer_input_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(transformer_input_dim, embedding_dim), nn.ReLU(inplace=True), nn.Dropout(p=0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)
        x = self.fc(x)
        return x


class AblatedCNNBackbone(nn.Module):
    """Pure CNN ablation backbone without Transformer."""

    def __init__(self, k: int, embedding_dim: int = 512) -> None:
        super().__init__()
        input_shape = 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(128, embedding_dim), nn.ReLU(inplace=True), nn.Dropout(p=0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MultiBranchNetwork(nn.Module):
    def __init__(
        self,
        k_value: int,
        backbone_embedding_dim: int,
        gc_dim: int,
        class_num: int = 3,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 512,
        options: ModelOptions | None = None,
    ) -> None:
        super().__init__()
        options = options or ModelOptions()
        self.options = options

        self.use_fcgr = options.use_fcgr
        self.use_gc = options.use_gc
        self.use_onehot = options.use_onehot

        if self.use_fcgr:
            if options.ablate_transformer:
                self.backbone = AblatedCNNBackbone(k=k_value, embedding_dim=backbone_embedding_dim)
            else:
                self.backbone = BackboneModel(
                    k=k_value,
                    embedding_dim=backbone_embedding_dim,
                    transformer_heads=transformer_heads,
                    transformer_layers=transformer_layers,
                    transformer_ff_dim=transformer_ff_dim,
                )
            self.rep_dim = backbone_embedding_dim
        else:
            self.backbone = None
            self.rep_dim = 0

        if self.use_onehot:
            self.one_d_branch_output_dim = 32
            self.one_d_branch = nn.Sequential(
                nn.Conv1d(in_channels=4, out_channels=64, kernel_size=8, padding="same"),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=8, padding=2),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, padding="same"),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=8, padding=3),
                nn.Conv1d(in_channels=128, out_channels=self.one_d_branch_output_dim, kernel_size=8, padding="same"),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            self.one_d_branch = None
            self.one_d_branch_output_dim = 0

        self.gc_dim_actual = gc_dim if self.use_gc else 0
        in_features_total = self.rep_dim + self.one_d_branch_output_dim + self.gc_dim_actual
        if in_features_total == 0:
            raise ValueError("At least one feature branch must be enabled.")
        self.classifier = nn.Sequential(
            nn.Linear(in_features_total, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, class_num),
        )

    def forward(self, x_fcgr: torch.Tensor, x_gc: torch.Tensor, x_onehot: torch.Tensor) -> torch.Tensor:
        features_to_cat = []
        if self.use_fcgr:
            features_to_cat.append(self.backbone(x_fcgr))
        if self.use_onehot:
            h_1d = self.one_d_branch(x_onehot.permute(0, 2, 1))
            features_to_cat.append(torch.flatten(h_1d, 1))
        if self.use_gc:
            features_to_cat.append(x_gc)
        combined_features = torch.cat(features_to_cat, dim=1)
        return self.classifier(combined_features)
