from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def count_wavs(dataset_root: Path) -> dict[str, object]:
    if not dataset_root.exists():
        return {
            "exists": False,
            "num_items": 0,
            "label_counts": {},
        }

    label_counts: dict[str, int] = {}
    for language_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        label_counts[language_dir.name] = sum(1 for _ in language_dir.rglob("*.wav"))

    return {
        "exists": True,
        "num_items": sum(label_counts.values()),
        "label_counts": label_counts,
    }


def get_torch_status() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {
            "installed": False,
            "error": str(exc),
        }

    cuda_available = torch.cuda.is_available()
    status: dict[str, object] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    if cuda_available:
        status["devices"] = [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(idx).total_memory / (1024**3),
                    2,
                ),
            }
            for idx in range(torch.cuda.device_count())
        ]
    return status


def get_huggingface_status() -> dict[str, object]:
    try:
        import huggingface_hub
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {
            "installed": False,
            "error": str(exc),
        }
    return {
        "installed": True,
        "version": huggingface_hub.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check cloud training readiness.")
    parser.add_argument("--dataset-root", default="", help="Optional dataset root to count.")
    parser.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root used for disk-space reporting.",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    disk = shutil.disk_usage(workspace_root)

    report = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "workspace_root": str(workspace_root),
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
        },
        "torch": get_torch_status(),
        "huggingface_hub": get_huggingface_status(),
    }

    if args.dataset_root:
        report["dataset"] = {
            "root": str(Path(args.dataset_root).expanduser().resolve()),
            **count_wavs(Path(args.dataset_root).expanduser()),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
