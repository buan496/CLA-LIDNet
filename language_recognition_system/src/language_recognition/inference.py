from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from .audio import load_wav, pad_or_trim
from .features import LogMelSpectrogram
from .model import CnnBiLstmAttentionModel


class LanguagePredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        self.device = torch.device(device)
        self.label_map: dict[str, int] = checkpoint["label_map"]
        self.id_to_label = {idx: label for label, idx in self.label_map.items()}
        self.config = checkpoint["config"]

        self.sample_rate = int(self.config.get("sample_rate", 16000))
        self.clip_duration = float(self.config.get("clip_duration", 3.0))
        self.n_mels = int(self.config.get("n_mels", 80))
        self.target_num_samples = int(self.sample_rate * self.clip_duration)
        self.featurizer = LogMelSpectrogram(sample_rate=self.sample_rate, n_mels=self.n_mels)

        self.model = CnnBiLstmAttentionModel(
            num_classes=len(self.label_map),
            n_mels=self.n_mels,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_file(self, audio_path: str | Path, top_k: int = 3) -> dict[str, Any]:
        samples, _ = load_wav(audio_path, target_sr=self.sample_rate)
        samples = pad_or_trim(samples, self.target_num_samples)
        waveform = torch.from_numpy(samples).to(self.device)
        features = self.featurizer(waveform).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(features)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        top_values, top_indices = torch.topk(probs, k=min(top_k, probs.numel()))
        top_predictions = [
            {
                "label": self.id_to_label[int(idx.item())],
                "score": float(val.item()),
            }
            for val, idx in zip(top_values, top_indices)
        ]

        return {
            "predicted_label": top_predictions[0]["label"],
            "predicted_score": top_predictions[0]["score"],
            "top_k": top_predictions,
        }

    def predict_batch(
        self,
        audio_dir: str | Path,
        output_csv: str | Path | None = None,
        top_k: int = 3,
        extensions: tuple[str, ...] = (".wav",),
    ) -> list[dict[str, Any]]:
        audio_dir = Path(audio_dir)
        audio_files = sorted(
            p for p in audio_dir.rglob("*") if p.suffix.lower() in extensions
        )

        results: list[dict[str, Any]] = []
        for audio_path in audio_files:
            try:
                pred = self.predict_file(audio_path, top_k=top_k)
                row: dict[str, Any] = {
                    "file": str(audio_path),
                    "predicted_label": pred["predicted_label"],
                    "predicted_score": pred["predicted_score"],
                }
                for rank, item in enumerate(pred["top_k"], start=1):
                    row[f"top{rank}_label"] = item["label"]
                    row[f"top{rank}_score"] = item["score"]
                results.append(row)
            except Exception as exc:
                results.append({"file": str(audio_path), "error": str(exc)})

        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            if results:
                fieldnames = list(results[0].keys())
                with output_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(results)

        return results
