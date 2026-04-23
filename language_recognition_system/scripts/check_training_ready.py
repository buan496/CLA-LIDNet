from __future__ import annotations

import argparse
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from language_recognition.audio import load_wav
from language_recognition.dataset import (
    discover_dataset,
    summarize_dataset,
    validate_dataset_for_training,
)
from language_recognition.features import LogMelSpectrogram
from language_recognition.model import CnnBiLstmAttentionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the project is ready for training.")
    parser.add_argument("--dataset-root", help="dataset/lang_name/*.wav")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    return parser.parse_args()


def build_tone_wav() -> str:
    sample_rate = 16000
    duration = 1.0
    time_axis = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = 0.2 * np.sin(2 * np.pi * 440 * time_axis)
    pcm = (samples * 32767).astype(np.int16)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp.close()
    with wave.open(temp.name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return temp.name


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "requested_device": args.device,
    }

    if args.device == "cuda" and not torch.cuda.is_available():
        report["device_warning"] = "请求使用 cuda，但当前环境中的 torch 未检测到可用 GPU。"

    tone_path = build_tone_wav()
    try:
        samples, sample_rate = load_wav(tone_path, target_sr=16000)
        features = LogMelSpectrogram(sample_rate=16000)(torch.from_numpy(samples))
        model_device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        model = CnnBiLstmAttentionModel(num_classes=3).to(model_device)
        logits, _ = model(features.unsqueeze(0).to(model_device))
        report["smoke_test"] = {
            "audio_load_ok": True,
            "sample_rate": sample_rate,
            "feature_shape": list(features.shape),
            "model_forward_shape": list(logits.shape),
        }
    finally:
        Path(tone_path).unlink(missing_ok=True)

    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        report["dataset_root"] = str(dataset_root)
        if not dataset_root.exists():
            report["dataset_status"] = "missing"
        else:
            items, _ = discover_dataset(dataset_root)
            summary = summarize_dataset(items)
            issues = validate_dataset_for_training(items)
            report["dataset_status"] = "ok" if not issues else "needs_attention"
            report["dataset_summary"] = {
                "num_items": summary.num_items,
                "num_labels": summary.num_labels,
                "label_counts": summary.label_counts,
            }
            report["dataset_issues"] = issues

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
