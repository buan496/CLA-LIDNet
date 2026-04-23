from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_MD = "论文材料/12-毕业论文正文_基于深度学习的语言识别系统_文献修订版.md"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
REF_ENTRY_RE = re.compile(r"^\[(\d+)\]\s+")


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "论文材料").exists() and (candidate / "language_recognition_system").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the current script path.")


def extract_section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start == -1:
        return ""
    trailing = text[start + len(marker) :]
    next_heading = trailing.find("\n## ")
    if next_heading == -1:
        return trailing.strip()
    return trailing[:next_heading].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit thesis structure and baseline completeness checks.")
    parser.add_argument("--input", default=DEFAULT_MD, help="Project-relative or absolute markdown path.")
    parser.add_argument("--min-references", type=int, default=20, help="Minimum expected numbered references.")
    parser.add_argument("--min-zh-abstract-chars", type=int, default=250, help="Minimum recommended Chinese abstract length.")
    parser.add_argument("--min-en-abstract-words", type=int, default=120, help="Minimum recommended English abstract word count.")
    return parser.parse_args()


def add_result(results: list[tuple[str, str]], level: str, message: str) -> None:
    results.append((level, message))


def main() -> None:
    args = parse_args()
    root = find_project_root(Path(__file__).resolve())
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path

    text = input_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    headings = [(idx + 1, len(m.group(1)), m.group(2).strip()) for idx, line in enumerate(lines) if (m := HEADING_RE.match(line))]
    results: list[tuple[str, str]] = []

    if headings and headings[0][1] == 1 and headings[0][2]:
        add_result(results, "OK", f"Found title: {headings[0][2]}")
    else:
        add_result(results, "ERROR", "Missing thesis title line beginning with '# '.")

    required_sections = ["摘要", "Abstract", "注释表", "参考文献", "附录"]
    for section in required_sections:
        if any(level == 2 and title == section for _, level, title in headings):
            add_result(results, "OK", f"Found required section: {section}")
        else:
            add_result(results, "ERROR", f"Missing required section: {section}")

    chapter_positions: list[int] = []
    for chapter_num in range(1, 7):
        match = next((i for i, (_, level, title) in enumerate(headings) if level == 2 and title.startswith(f"第{chapter_num}章")), None)
        if match is None:
            add_result(results, "ERROR", f"Missing chapter heading: 第{chapter_num}章")
        else:
            chapter_positions.append(match)
            add_result(results, "OK", f"Found chapter heading: 第{chapter_num}章")

    if chapter_positions and chapter_positions != sorted(chapter_positions):
        add_result(results, "ERROR", "Chapter order is not monotonic from 第1章 to 第6章.")

    if "关键词：" in text:
        add_result(results, "OK", "Found Chinese keywords line.")
    else:
        add_result(results, "ERROR", "Missing Chinese keywords line '关键词：'.")

    if "Key words:" in text or "Keywords:" in text:
        add_result(results, "OK", "Found English keywords line.")
    else:
        add_result(results, "ERROR", "Missing English keywords line such as 'Key words:'.")

    zh_abstract = extract_section(text, "摘要")
    zh_abstract_chars = len(re.sub(r"\s+", "", zh_abstract))
    if zh_abstract_chars >= args.min_zh_abstract_chars:
        add_result(results, "OK", f"Chinese abstract length looks adequate: {zh_abstract_chars} chars.")
    else:
        add_result(results, "WARN", f"Chinese abstract may be short: {zh_abstract_chars} chars, recommended >= {args.min_zh_abstract_chars}.")

    en_abstract = extract_section(text, "Abstract")
    en_abstract_words = len(re.findall(r"[A-Za-z0-9\-]+", en_abstract))
    if en_abstract_words >= args.min_en_abstract_words:
        add_result(results, "OK", f"English abstract length looks adequate: {en_abstract_words} words.")
    else:
        add_result(results, "WARN", f"English abstract may be short: {en_abstract_words} words, recommended >= {args.min_en_abstract_words}.")

    ref_section = extract_section(text, "参考文献")
    ref_count = sum(1 for line in ref_section.splitlines() if REF_ENTRY_RE.match(line.strip()))
    if ref_count >= args.min_references:
        add_result(results, "OK", f"Reference count meets baseline: {ref_count}.")
    else:
        add_result(results, "WARN", f"Reference count below baseline: {ref_count}, recommended >= {args.min_references}.")

    chapter_titles = [title for _, level, title in headings if level == 2 and title.startswith("第")]
    for chapter_title in chapter_titles:
        section_text = extract_section(text, chapter_title)
        subsection_count = len(re.findall(r"^###\s+", section_text, flags=re.MULTILINE))
        if subsection_count == 0:
            add_result(results, "WARN", f"{chapter_title} currently has no level-3 subsections.")

    error_count = sum(1 for level, _ in results if level == "ERROR")
    warn_count = sum(1 for level, _ in results if level == "WARN")

    print(f"input={input_path}")
    print(f"errors={error_count}")
    print(f"warnings={warn_count}")
    print(f"status={'ok' if error_count == 0 else 'needs_attention'}")
    print()
    for level, message in results:
        print(f"[{level}] {message}")

    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command-line reporting
        print(f"error={exc}", file=sys.stderr)
        raise
