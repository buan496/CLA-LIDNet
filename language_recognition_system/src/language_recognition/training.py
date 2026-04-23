from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import (
    AugmentConfig,
    LanguageDataset,
    discover_dataset,
    stratified_split,
    summarize_dataset,
    validate_dataset_for_training,
)
from .features import LogMelSpectrogram
from .metrics import accuracy_score, macro_f1_score
from .model import CnnBiLstmAttentionModel


@dataclass
class TrainConfig:
    dataset_root: str
    output_dir: str
    sample_rate: int = 16000
    clip_duration: float = 3.0
    n_mels: int = 80
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    test_size: float = 0.2
    val_size: float = 0.2
    random_seed: int = 42
    device: str = "cpu"
    train_repeat_factor: int = 3
    enable_waveform_augment: bool = True
    enable_spec_augment: bool = True
    label_smoothing: float = 0.05
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 8
    min_epochs: int = 12
    num_workers: int = 0
    pin_memory: bool = True


def _build_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> list[list[int]]:
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t][p] += 1
    return matrix


def _run_epoch(
    model: CnnBiLstmAttentionModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: AdamW | None = None,
    grad_clip_norm: float = 0.0,
) -> dict[str, Any]:
    model.train(optimizer is not None)

    total_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []

    with torch.set_grad_enabled(optimizer is not None):
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits, _ = model(features)
            loss = criterion(logits, targets)

            if optimizer is not None:
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * targets.size(0)
            preds = torch.argmax(logits, dim=1)
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy_score(all_targets, all_preds),
        "macro_f1": macro_f1_score(all_targets, all_preds, num_classes),
        "_targets": all_targets,
        "_preds": all_preds,
    }


def train_model(config: TrainConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = str(config.device).lower()
    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    items, label_map = discover_dataset(config.dataset_root)
    issues = validate_dataset_for_training(items)
    if issues:
        raise ValueError("数据集暂时不适合训练：\n- " + "\n- ".join(issues))

    dataset_summary = summarize_dataset(items)
    train_items, test_items = stratified_split(items, config.test_size, config.random_seed)
    train_items, val_items = stratified_split(train_items, config.val_size, config.random_seed + 1)

    if not train_items:
        raise ValueError("训练集为空，请检查数据划分比例。")
    if not val_items:
        raise ValueError("验证集为空，请增加样本量或调小 val_size。")
    if not test_items:
        raise ValueError("测试集为空，请增加样本量或调小 test_size。")

    featurizer = LogMelSpectrogram(sample_rate=config.sample_rate, n_mels=config.n_mels)
    train_augment_config = AugmentConfig(
        enabled=config.enable_waveform_augment or config.enable_spec_augment,
        repeat_factor=max(1, config.train_repeat_factor),
        gain_prob=0.5 if config.enable_waveform_augment else 0.0,
        noise_prob=0.55 if config.enable_waveform_augment else 0.0,
        shift_prob=0.5 if config.enable_waveform_augment else 0.0,
        speed_prob=0.35 if config.enable_waveform_augment else 0.0,
        spec_augment_prob=0.7 if config.enable_spec_augment else 0.0,
    )
    train_dataset = LanguageDataset(
        train_items,
        config.sample_rate,
        config.clip_duration,
        featurizer,
        training=True,
        augment_config=train_augment_config,
        random_seed=config.random_seed,
    )
    val_dataset = LanguageDataset(
        val_items,
        config.sample_rate,
        config.clip_duration,
        featurizer,
        training=False,
        random_seed=config.random_seed + 1,
    )
    test_dataset = LanguageDataset(
        test_items,
        config.sample_rate,
        config.clip_duration,
        featurizer,
        training=False,
        random_seed=config.random_seed + 2,
    )

    if config.num_workers < 0:
        cpu_count = os.cpu_count() or 2
        data_workers = max(0, min(8, cpu_count // 2))
    else:
        data_workers = config.num_workers

    train_loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": data_workers,
        "pin_memory": bool(config.pin_memory and str(config.device).lower().startswith("cuda")),
    }
    if data_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2

    eval_loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": 0,
        "pin_memory": bool(config.pin_memory and str(config.device).lower().startswith("cuda")),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **eval_loader_kwargs)

    device = torch.device(config.device)
    model = CnnBiLstmAttentionModel(num_classes=len(label_map), n_mels=config.n_mels).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=max(0.0, config.label_smoothing))
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01)

    history: list[dict[str, float | int]] = []
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_checkpoint = output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            len(label_map),
            optimizer,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_metrics = _run_epoch(model, val_loader, criterion, device, len(label_map))
        scheduler.step()

        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"lr={epoch_record['lr']:.2e} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map": label_map,
                    "config": asdict(config),
                },
                best_checkpoint,
            )
        else:
            epochs_without_improvement += 1

        if (
            config.early_stop_patience > 0
            and epoch >= config.min_epochs
            and epochs_without_improvement >= config.early_stop_patience
        ):
            print(
                f"Early stopping at epoch {epoch:03d}: "
                f"no validation F1 improvement for {epochs_without_improvement} epochs."
            )
            break

    saved = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    test_raw = _run_epoch(model, test_loader, criterion, device, len(label_map))

    id_to_label = {idx: name for name, idx in label_map.items()}
    num_classes = len(label_map)
    confusion = _build_confusion_matrix(test_raw["_targets"], test_raw["_preds"], num_classes)
    label_names = [id_to_label[i] for i in range(num_classes)]

    test_metrics = {
        "loss": test_raw["loss"],
        "accuracy": test_raw["accuracy"],
        "macro_f1": test_raw["macro_f1"],
    }

    summary: dict[str, Any] = {
        "best_checkpoint": str(best_checkpoint),
        "label_map": label_map,
        "dataset_summary": {
            "num_items": dataset_summary.num_items,
            "num_labels": dataset_summary.num_labels,
            "label_counts": dataset_summary.label_counts,
        },
        "train_size": len(train_items),
        "effective_train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "best_epoch": best_epoch,
        "trained_epochs": len(history),
        "early_stopped": len(history) < config.epochs,
        "augmentation": {
            "enabled": train_augment_config.enabled,
            "repeat_factor": train_augment_config.repeat_factor,
            "waveform_augment": config.enable_waveform_augment,
            "spec_augment": config.enable_spec_augment,
        },
        "runtime": {
            "device": str(config.device),
            "num_workers": data_workers,
            "pin_memory": bool(train_loader_kwargs["pin_memory"]),
            "eval_num_workers": 0,
        },
        "test_metrics": test_metrics,
        "confusion_matrix": {
            "labels": label_names,
            "matrix": confusion,
        },
        "history": history,
    }

    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return summary
