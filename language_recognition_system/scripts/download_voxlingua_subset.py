from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a VoxLingua107 subset and export it to dataset/lang/*.wav layout."
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output dataset root, e.g. H:\\deepmode\\dataset_vox_trial",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "ja", "zh"],
        help="VoxLingua107 language codes to export.",
    )
    parser.add_argument(
        "--max-shards-per-language",
        type=int,
        default=2,
        help="Maximum training tar shards to download for each language.",
    )
    parser.add_argument(
        "--max-files-per-language",
        type=int,
        default=0,
        help="Optional maximum number of wav files to export per language. 0 means no file cap.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def list_language_shards(repo_id: str, language: str) -> list[str]:
    api = HfApi()
    prefix = f"train/{language}/"
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return sorted([name for name in repo_files if name.startswith(prefix) and name.endswith(".tar")])


def export_language_subset(
    repo_id: str,
    language: str,
    output_root: Path,
    max_shards_per_language: int,
    max_files_per_language: int,
) -> dict[str, object]:
    shard_names = list_language_shards(repo_id=repo_id, language=language)
    selected_shards = shard_names[: max(0, max_shards_per_language)]
    export_dir = output_root / language
    export_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    shard_reports: list[dict[str, object]] = []

    for shard_name in selected_shards:
        shard_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=shard_name)
        shard_stem = Path(shard_name).stem
        shard_exported = 0

        with tarfile.open(shard_path, "r") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                if max_files_per_language > 0 and exported >= max_files_per_language:
                    break

                source = archive.extractfile(member)
                if source is None:
                    continue

                target_name = f"vox_{language}_{shard_stem}_{Path(member.name).name}"
                target_path = export_dir / target_name
                with source, target_path.open("wb") as destination:
                    destination.write(source.read())

                exported += 1
                shard_exported += 1

        shard_reports.append(
            {
                "shard": shard_name,
                "exported_wavs": shard_exported,
            }
        )

        if max_files_per_language > 0 and exported >= max_files_per_language:
            break

    return {
        "available_shards": len(shard_names),
        "selected_shards": selected_shards,
        "exported_wavs": exported,
        "shards": shard_reports,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    repo_id = "TalTechNLP/voxlingua107_wds"
    report: dict[str, object] = {
        "dataset": repo_id,
        "output_root": str(output_root),
        "languages": args.languages,
        "max_shards_per_language": args.max_shards_per_language,
        "max_files_per_language": args.max_files_per_language,
        "exported": {},
    }

    for language in args.languages:
        print(f"[INFO] exporting language: {language}")
        report["exported"][language] = export_language_subset(
            repo_id=repo_id,
            language=language,
            output_root=output_root,
            max_shards_per_language=args.max_shards_per_language,
            max_files_per_language=args.max_files_per_language,
        )
        print(
            f"[INFO] exported {report['exported'][language]['exported_wavs']} wav files for {language}"
        )

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
