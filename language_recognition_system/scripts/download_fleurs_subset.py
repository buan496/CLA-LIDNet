from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a small FLEURS subset and export it to dataset/lang/*.wav layout."
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output dataset root, for example H:\\deepmode\\dataset",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["cmn_hans_cn", "en_us", "ja_jp"],
        help="FLEURS language configs to download.",
    )
    parser.add_argument(
        "--config-map",
        default="cmn_hans_cn:zh en_us:en ja_jp:ja",
        help="Mapping from FLEURS config name to output folder name.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        help="FLEURS splits to export.",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=240,
        help="Maximum number of samples to export per language per split.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def parse_mapping(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in raw.split():
        if ":" not in item:
            continue
        source, target = item.split(":", 1)
        mapping[source] = target
    return mapping


def read_fleurs_filenames(tsv_path: str | Path, max_per_split: int) -> list[str]:
    filenames: list[str] = []
    with Path(tsv_path).open("r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            filenames.append(row[1])
            if len(filenames) >= max_per_split:
                break
    return filenames


def export_split(
    config_name: str,
    lang_code: str,
    split_name: str,
    output_root: Path,
    max_per_split: int,
) -> int:
    tsv_name = f"data/{config_name}/{split_name}.tsv"
    archive_name = f"data/{config_name}/audio/{split_name}.tar.gz"

    tsv_path = hf_hub_download(repo_id="google/fleurs", repo_type="dataset", filename=tsv_name)
    archive_path = hf_hub_download(repo_id="google/fleurs", repo_type="dataset", filename=archive_name)

    wanted_filenames = read_fleurs_filenames(tsv_path, max_per_split)
    wanted_set = set(wanted_filenames)
    exported = 0

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_name = Path(member.name).name
            if member_name not in wanted_set:
                continue

            target_name = f"{split_name}_{lang_code}_{member_name}"
            target_path = output_root / lang_code / target_name
            target_path.parent.mkdir(parents=True, exist_ok=True)

            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)

            exported += 1
            if exported >= max_per_split:
                break

    return exported


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_map = parse_mapping(args.config_map)

    report: dict[str, object] = {
        "dataset": "google/fleurs",
        "output_root": str(output_root),
        "configs": args.configs,
        "splits": args.splits,
        "max_per_split": args.max_per_split,
        "exported": {},
    }

    for config_name in args.configs:
        lang_code = config_map.get(config_name, config_name)
        report["exported"][lang_code] = {}
        for split_name in args.splits:
            count = export_split(
                config_name=config_name,
                lang_code=lang_code,
                split_name=split_name,
                output_root=output_root,
                max_per_split=args.max_per_split,
            )
            report["exported"][lang_code][split_name] = count
            print(f"[INFO] exported {count} files for {config_name} / {split_name}")

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
