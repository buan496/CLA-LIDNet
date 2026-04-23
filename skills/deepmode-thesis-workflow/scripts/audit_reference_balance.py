from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_MD = "论文材料/12-毕业论文正文_基于深度学习的语言识别系统_文献修订版.md"
CJK_RE = re.compile(r"[\u3400-\u9fff]")
REF_SECTION_RE = re.compile(r"^##\s+参考文献\s*$", re.MULTILINE)
REF_ENTRY_RE = re.compile(r"^\[(\d+)\]\s+(.*)$")


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "论文材料").exists() and (candidate / "language_recognition_system").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the current script path.")


def extract_reference_lines(markdown_text: str) -> list[str]:
    match = REF_SECTION_RE.search(markdown_text)
    if not match:
        raise ValueError("Could not find a '## 参考文献' section in the thesis markdown.")

    trailing = markdown_text[match.end() :].strip("\n")
    cutoff = len(trailing)
    for marker in ("\n---", "\n## "):
        index = trailing.find(marker)
        if index != -1:
            cutoff = min(cutoff, index)
    section = trailing[:cutoff].strip()
    return [line.strip() for line in section.splitlines() if line.strip()]


def classify_reference(entry_text: str) -> str:
    return "chinese" if CJK_RE.search(entry_text) else "non_chinese"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Chinese-vs-non-Chinese reference balance in the thesis markdown.")
    parser.add_argument("--input", default=DEFAULT_MD, help="Project-relative or absolute markdown path.")
    parser.add_argument("--min-chinese-ratio", type=float, default=0.33, help="Minimum desired Chinese reference ratio.")
    parser.add_argument("--show-entries", action="store_true", help="Print each parsed reference with its classification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root(Path(__file__).resolve())
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path

    text = input_path.read_text(encoding="utf-8")
    lines = extract_reference_lines(text)
    entries: list[tuple[str, str]] = []
    for line in lines:
        match = REF_ENTRY_RE.match(line)
        if match:
            entry_text = match.group(2).strip()
            entries.append((line, classify_reference(entry_text)))

    total = len(entries)
    chinese = sum(1 for _, kind in entries if kind == "chinese")
    non_chinese = total - chinese
    ratio = (chinese / total) if total else 0.0

    print(f"input={input_path}")
    print(f"total_references={total}")
    print(f"chinese_references={chinese}")
    print(f"non_chinese_references={non_chinese}")
    print(f"chinese_ratio={ratio:.3f}")
    print(f"target_ratio={args.min_chinese_ratio:.3f}")

    if total == 0:
        print("status=error")
        print("message=No numbered references were parsed from the 参考文献 section.")
        raise SystemExit(1)

    if ratio < args.min_chinese_ratio:
        print("status=needs_more_chinese_references")
    else:
        print("status=ok")

    if args.show_entries:
        print("\nentries:")
        for line, kind in entries:
            print(f"[{kind}] {line}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command-line reporting
        print(f"error={exc}", file=sys.stderr)
        raise
