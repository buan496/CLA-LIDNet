from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class CnnBiLstmAttentionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_mels: int = 80,
        hidden_size: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
        )
        reduced_mels = n_mels // 4
        encoder_input_size = 64 * reduced_mels

        self.encoder = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pooling = AttentionPooling(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = features.unsqueeze(1)
        x = self.feature_extractor(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        batch_size, time_steps, channels, mel_bins = x.shape
        x = x.view(batch_size, time_steps, channels * mel_bins)
        encoded, _ = self.encoder(x)
        pooled, attention_weights = self.pooling(encoded)
        logits = self.classifier(pooled)
        return logits, attention_weights
