from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert the CommonLanguage dataset into dataset/lang/*.wav layout."
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output dataset root, e.g. H:\\deepmode\\dataset",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["Chinese_China", "English", "Japanese"],
        help="Language names as defined in the CommonLanguage dataset metadata.",
    )
    parser.add_argument(
        "--language-map",
        default="Chinese_China:zh English:en Japanese:ja",
        help="Mapping from dataset language names to output folder names. Format: 'Chinese_China:zh English:en'",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=200,
        help="Maximum number of samples per language per split.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Reserved for compatibility. Audio is exported as original wav and resampled during training.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to export.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def load_hf_dependencies():
    try:
        from datasets import Audio, load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing datasets dependency. Run: conda run -n nlp100 python -m pip install datasets"
        ) from exc
    return Audio, load_dataset


def parse_language_map(raw_mapping: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in raw_mapping.split():
        if ":" not in item:
            continue
        source, target = item.split(":", 1)
        mapping[source] = target
    return mapping


def detect_language_column(example: dict[str, Any]) -> str:
    candidates = [
        "language",
        "lang",
        "locale",
        "language_name",
        "sentence_language",
    ]
    for candidate in candidates:
        if candidate in example:
            return candidate
    raise KeyError(f"Unable to detect language column. Available fields: {list(example.keys())}")


def list_parquet_files_for_split(repo_id: str, split_name: str, revision: str) -> list[str]:
    api = HfApi()
    prefix = f"full/{split_name}/"
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    return sorted([name for name in repo_files if name.startswith(prefix) and name.endswith(".parquet")])


def load_commonlanguage_split(split_name: str):
    Audio, load_dataset = load_hf_dependencies()
    repo_id = "speechbrain/common_language"
    revision = "refs/convert/parquet"
    parquet_files = list_parquet_files_for_split(repo_id=repo_id, split_name=split_name, revision=revision)
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found for split: {split_name}")

    local_paths = [
        hf_hub_download(repo_id=repo_id, repo_type="dataset", revision=revision, filename=filename)
        for filename in parquet_files
    ]

    dataset = load_dataset("parquet", data_files={split_name: local_paths}, split=split_name)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def decode_language_name(dataset, value: Any) -> str:
    feature = dataset.features.get("language")
    if feature is not None and hasattr(feature, "int2str") and isinstance(value, int):
        return str(feature.int2str(value))
    return str(value)


def resolve_audio_bytes(row: dict[str, Any]) -> bytes:
    audio_info = row.get("audio")
    if isinstance(audio_info, dict):
        raw_bytes = audio_info.get("bytes")
        if raw_bytes:
            return raw_bytes

        audio_path = audio_info.get("path")
        if audio_path:
            audio_path = Path(str(audio_path))
            if audio_path.exists():
                return audio_path.read_bytes()

    row_path = row.get("path")
    if row_path:
        source_path = Path(str(row_path))
        if source_path.exists():
            return source_path.read_bytes()

    raise ValueError("unable to resolve audio bytes from row")


def export_audio_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    language_map = parse_language_map(args.language_map)
    selected_languages = set(args.languages)
    report: dict[str, Any] = {
        "dataset": "speechbrain/common_language",
        "source_revision": "refs/convert/parquet",
        "output_root": str(output_root),
        "sample_rate": args.sample_rate,
        "splits": args.splits,
        "languages": args.languages,
        "exported": {},
    }

    for split_name in args.splits:
        print(f"[INFO] loading split: {split_name}")
        dataset = load_commonlanguage_split(split_name=split_name)

        if len(dataset) == 0:
            print(f"[WARN] split {split_name} is empty, skipped.")
            continue

        example = dataset[0]
        language_column = detect_language_column(example)
        split_counts: dict[str, int] = {}

        for row in dataset:
            language_name = decode_language_name(dataset, row[language_column])
            if language_name not in selected_languages:
                continue

            split_counts.setdefault(language_name, 0)
            if split_counts[language_name] >= args.max_per_split:
                continue

            mapped_name = language_map.get(language_name, language_name.lower())
            audio_bytes = resolve_audio_bytes(row)
            file_index = split_counts[language_name]
            file_name = f"{split_name}_{mapped_name}_{file_index:05d}.wav"
            export_path = output_root / mapped_name / file_name
            export_audio_file(export_path, audio_bytes)
            split_counts[language_name] += 1

        report["exported"][split_name] = split_counts
        print(f"[INFO] split {split_name} exported: {json.dumps(split_counts, ensure_ascii=False)}")

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
