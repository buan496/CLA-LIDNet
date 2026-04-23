from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MD = "论文材料/12-毕业论文正文_基于深度学习的语言识别系统_文献修订版.md"
DEFAULT_DOCX = "论文材料/12-毕业论文正文_基于深度学习的语言识别系统_文献修订版.docx"


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "论文材料").exists() and (candidate / "language_recognition_system").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the current script path.")


def run_step(command: list[str], dry_run: bool) -> None:
    rendered = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"$ {rendered}")
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate thesis figures and/or DOCX outputs for the Deepmode thesis project.")
    parser.add_argument("--input", default=DEFAULT_MD, help="Project-relative or absolute markdown path.")
    parser.add_argument("--output", default=DEFAULT_DOCX, help="Project-relative or absolute DOCX path.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure regeneration.")
    parser.add_argument("--skip-docx", action="store_true", help="Skip DOCX export.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child scripts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    root = find_project_root(Path(__file__).resolve())
    python_exe = str(Path(args.python))

    paper_dir = root / "论文材料"
    figure_script = paper_dir / "tools" / "generate_paper_assets.py"
    export_script = paper_dir / "tools" / "export_md_to_docx.py"

    input_md = resolve_path(root, args.input)
    output_docx = resolve_path(root, args.output)

    if not args.skip_figures:
        run_step([python_exe, str(figure_script)], dry_run=args.dry_run)

    if not args.skip_docx:
        run_step(
            [python_exe, str(export_script), "--input", str(input_md), "--output", str(output_docx)],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
