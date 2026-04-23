from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from language_recognition.training import TrainConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a language recognition model.")
    parser.add_argument("--dataset-root", required=True, help="dataset/lang_name/*.wav")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints" / "run1"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-duration", type=float, default=3.0)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-repeat-factor", type=int, default=3)
    parser.add_argument("--disable-waveform-augment", action="store_true")
    parser.add_argument("--disable-spec-augment", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=-1, help="DataLoader workers; -1 picks a small automatic value.")
    parser.add_argument("--disable-pin-memory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        n_mels=args.n_mels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.random_seed,
        device=args.device,
        train_repeat_factor=args.train_repeat_factor,
        enable_waveform_augment=not args.disable_waveform_augment,
        enable_spec_augment=not args.disable_spec_augment,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        min_epochs=args.min_epochs,
        num_workers=args.num_workers,
        pin_memory=not args.disable_pin_memory,
    )
    result = train_model(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
