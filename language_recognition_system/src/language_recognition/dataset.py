from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .audio import load_wav, pad_or_trim
from .features import LogMelSpectrogram


@dataclass
class DatasetItem:
    path: str
    label_name: str
    label_id: int


@dataclass
class DatasetSummary:
    num_items: int
    num_labels: int
    label_counts: dict[str, int]


@dataclass
class AugmentConfig:
    enabled: bool = False
    repeat_factor: int = 1
    gain_prob: float = 0.5
    noise_prob: float = 0.55
    shift_prob: float = 0.5
    speed_prob: float = 0.35
    spec_augment_prob: float = 0.7
    max_shift_ratio: float = 0.12
    speed_min_rate: float = 0.92
    speed_max_rate: float = 1.08
    max_freq_mask_width: int = 10
    max_time_mask_width: int = 14


def discover_dataset(dataset_root: str | Path) -> tuple[List[DatasetItem], dict[str, int]]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    language_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
    label_map = {directory.name: idx for idx, directory in enumerate(language_dirs)}
    items: List[DatasetItem] = []

    for directory in language_dirs:
        label_id = label_map[directory.name]
        for audio_path in sorted(directory.rglob("*.wav")):
            items.append(
                DatasetItem(
                    path=str(audio_path),
                    label_name=directory.name,
                    label_id=label_id,
                )
            )

    if not items:
        raise ValueError(f"no wav files found under {root}")

    return items, label_map


def summarize_dataset(items: List[DatasetItem]) -> DatasetSummary:
    label_counts: dict[str, int] = {}
    for item in items:
        label_counts[item.label_name] = label_counts.get(item.label_name, 0) + 1
    return DatasetSummary(
        num_items=len(items),
        num_labels=len(label_counts),
        label_counts=dict(sorted(label_counts.items(), key=lambda pair: pair[0])),
    )


def validate_dataset_for_training(
    items: List[DatasetItem],
    min_labels: int = 2,
    min_samples_per_label: int = 3,
) -> list[str]:
    issues: list[str] = []
    summary = summarize_dataset(items)

    if summary.num_labels < min_labels:
        issues.append(f"至少需要 {min_labels} 个语言类别，当前只有 {summary.num_labels} 个。")

    for label_name, count in summary.label_counts.items():
        if count < min_samples_per_label:
            issues.append(
                f"语言类别 '{label_name}' 只有 {count} 条样本，建议至少准备 {min_samples_per_label} 条。"
            )

    return issues


def stratified_split(
    items: List[DatasetItem],
    split_ratio: float,
    random_seed: int,
) -> tuple[List[DatasetItem], List[DatasetItem]]:
    rng = random.Random(random_seed)
    grouped: dict[int, List[DatasetItem]] = {}
    for item in items:
        grouped.setdefault(item.label_id, []).append(item)

    left: List[DatasetItem] = []
    right: List[DatasetItem] = []
    for label_items in grouped.values():
        shuffled = label_items[:]
        rng.shuffle(shuffled)
        split_index = max(1, int(len(shuffled) * split_ratio))
        if split_index >= len(shuffled):
            split_index = len(shuffled) - 1
        left.extend(shuffled[split_index:])
        right.extend(shuffled[:split_index])

    rng.shuffle(left)
    rng.shuffle(right)
    return left, right


def random_pad_or_trim(
    samples,
    target_num_samples: int,
    rng: random.Random,
):
    if samples.shape[0] == target_num_samples:
        return samples
    if samples.shape[0] > target_num_samples:
        start = rng.randint(0, samples.shape[0] - target_num_samples)
        return samples[start : start + target_num_samples]

    padded = samples.new_zeros(target_num_samples)
    offset = rng.randint(0, target_num_samples - samples.shape[0])
    padded[offset : offset + samples.shape[0]] = samples
    return padded


class LanguageDataset(Dataset):
    def __init__(
        self,
        items: List[DatasetItem],
        sample_rate: int = 16000,
        clip_duration: float = 3.0,
        featurizer: LogMelSpectrogram | None = None,
        training: bool = False,
        augment_config: AugmentConfig | None = None,
        random_seed: int = 42,
    ) -> None:
        self.items = items
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.target_num_samples = int(sample_rate * clip_duration)
        self.featurizer = featurizer or LogMelSpectrogram(sample_rate=sample_rate)
        self.training = training
        self.augment_config = augment_config or AugmentConfig()
        self.repeat_factor = max(
            1,
            self.augment_config.repeat_factor if self.training and self.augment_config.enabled else 1,
        )
        self.random_seed = random_seed
        self.access_counter = 0

    def __len__(self) -> int:
        return len(self.items) * self.repeat_factor

    def _make_rng(self, index: int) -> random.Random:
        self.access_counter += 1
        return random.Random(self.random_seed + index * 1_000_003 + self.access_counter)

    def _augment_waveform(self, waveform: torch.Tensor, rng: random.Random) -> torch.Tensor:
        config = self.augment_config

        if rng.random() < config.speed_prob:
            rate = rng.uniform(config.speed_min_rate, config.speed_max_rate)
            target_length = max(1, int(round(waveform.numel() / rate)))
            waveform = F.interpolate(
                waveform.view(1, 1, -1),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).view(-1)

        waveform = random_pad_or_trim(waveform, self.target_num_samples, rng)

        if rng.random() < config.shift_prob:
            max_shift = max(1, int(self.target_num_samples * config.max_shift_ratio))
            shift = rng.randint(-max_shift, max_shift)
            if shift != 0:
                waveform = torch.roll(waveform, shifts=shift, dims=0)
                if shift > 0:
                    waveform[:shift] = 0.0
                else:
                    waveform[shift:] = 0.0

        if rng.random() < config.gain_prob:
            gain = 10 ** (rng.uniform(-4.0, 4.0) / 20.0)
            waveform = waveform * gain

        if rng.random() < config.noise_prob:
            noise_generator = torch.Generator()
            noise_generator.manual_seed(rng.randint(0, 2**31 - 1))
            noise = torch.randn(waveform.shape, generator=noise_generator, dtype=waveform.dtype)
            signal_rms = waveform.pow(2).mean().sqrt().clamp(min=1e-5)
            noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-5)
            snr_db = rng.uniform(18.0, 32.0)
            noise_scale = signal_rms / (10 ** (snr_db / 20.0) * noise_rms)
            waveform = waveform + noise * noise_scale

        return waveform.clamp(-1.0, 1.0)

    def _augment_features(self, features: torch.Tensor, rng: random.Random) -> torch.Tensor:
        config = self.augment_config
        if rng.random() >= config.spec_augment_prob:
            return features

        augmented = features.clone()
        num_mels, time_steps = augmented.shape

        max_freq_width = min(config.max_freq_mask_width, max(1, num_mels // 6))
        max_time_width = min(config.max_time_mask_width, max(1, time_steps // 8))

        if max_freq_width > 1:
            freq_width = rng.randint(1, max_freq_width)
            freq_start = rng.randint(0, max(0, num_mels - freq_width))
            augmented[freq_start : freq_start + freq_width, :] = 0.0

        if max_time_width > 1:
            time_width = rng.randint(1, max_time_width)
            time_start = rng.randint(0, max(0, time_steps - time_width))
            augmented[:, time_start : time_start + time_width] = 0.0

        return augmented

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index % len(self.items)]
        samples, _ = load_wav(item.path, target_sr=self.sample_rate)
        rng = self._make_rng(index)

        if self.training and self.augment_config.enabled:
            waveform = torch.from_numpy(samples.copy())
            waveform = self._augment_waveform(waveform, rng)
        else:
            samples = pad_or_trim(samples, self.target_num_samples)
            waveform = torch.from_numpy(samples)

        features = self.featurizer(waveform)
        if self.training and self.augment_config.enabled:
            features = self._augment_features(features, rng)
        return features, torch.tensor(item.label_id, dtype=torch.long)
