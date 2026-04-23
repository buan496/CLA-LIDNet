# Project Tooling

## Existing thesis-related tools

| Tool | Path | Current role | Status | Update priority |
|---|---|---|---|---|
| Thesis markdown | `../../论文材料/12-毕业论文正文_基于深度学习的语言识别系统_文献修订版.md` | Main thesis source of truth | In use | High |
| Requirement checklist | `../../论文材料/0-排版与提交规范清单.md` | Local formatting and submission rules | In use | Low |
| Task book draft | `../../论文材料/1-毕业论文任务书_草稿.md` | Scope and deliverable anchor | In use | Low |
| Literature review draft | `../../论文材料/3-文献综述_草稿.md` | Early background material | In use | Low |
| DOCX exporter | `../../论文材料/tools/export_md_to_docx.py` | Convert thesis markdown into `.docx` | Works, but basic | High |
| Figure generator | `../../论文材料/tools/generate_paper_assets.py` | Generate SVG and PNG thesis figures | Works, but needs refinement | High |
| Structure audit | `../scripts/audit_thesis_structure.py` | Check hard structure/content baseline requirements | New | High |
| Figure outputs | `../../论文材料/figures/*` | Reusable paper figures | In use | Medium |
| Training entry script | `../../language_recognition_system/scripts/train.py` | Run experiments and create training summaries | In use | Medium |
| CommonLanguage downloader | `../../language_recognition_system/scripts/download_commonlanguage.py` | Export a three-language CommonLanguage subset into `dataset/lang/*.wav` | Updated for parquet path | Medium |
| VoxLingua subset downloader | `../../language_recognition_system/scripts/download_voxlingua_subset.py` | Export shard-limited VoxLingua107 subsets for trial scaling runs | New | High |
| Prediction script | `../../language_recognition_system/scripts/predict.py` | Single-file inference for demos | In use | Low |
| Training readiness check | `../../language_recognition_system/scripts/check_training_ready.py` | Smoke test environment, dataset, and model path | In use | Low |
| Experiment summary | `../../language_recognition_system/checkpoints/run1/train_summary.json` | Real metrics, confusion matrix, and history for the thesis | In use | Medium |
| System README | `../../language_recognition_system/README.md` | Canonical runbook for data, training, inference, and demo | In use | Medium |
| Template extraction text | `../../template_extracted.txt` | Reference text extracted from a school-style sample PDF | Reference only | Low |

## Immediate upgrade priorities

Important framing:

- No current tool can automatically generate a qualified thesis.
- Current tools only support auditing, exporting, regenerating figures, and reusing experiment outputs.
- Thesis structure and content quality must be guaranteed by deliberate editing and verification.

### 1. `export_md_to_docx.py`

Current strengths:

- Sets page size and margins.
- Applies basic Chinese and English fonts.
- Handles headings, lists, tables, code blocks, and image insertion.

Current gaps:

- Builds a fresh document instead of aligning to a school template.
- Does not manage cover pages, directory pages, page numbers, or table of contents.
- Uses a minimal markdown parser and ignores equations, cross-references, and richer reference formatting.
- Uses fixed image width and simple captions, which is weak for thesis-grade layout control.
- Hard-codes project paths, which makes repeated reuse less flexible.

Upgrade direction:

- Parameterize paths and output names.
- Add template-aware styles or a template-based export mode.
- Improve title, abstract, keywords, captions, and bibliography formatting.
- Add more reliable handling for equations, appendix structure, and reference sections.

### 2. `generate_paper_assets.py`

Current strengths:

- Produces editable SVG plus exportable PNG.
- Reuses `train_summary.json` for curve and confusion matrix generation.
- Already covers the thesis's main conceptual diagrams.

Current gaps:

- `add_box()` and `add_small_box()` accept a `fill` argument but still render boxes as white, which means the intended color system is not actually applied.
- Arrow size, marker size, and label size are globally small for thesis printing and later PPT reuse.
- Layouts are mostly hand-tuned, not parameterized for denser annotations or slide-friendly variants.
- There is no explicit mode for 16:9 slide reuse, transparent export, or higher-density explanatory labels.

Upgrade direction:

- Fix the `fill` usage bug first.
- Add global controls for arrow thickness, marker size, title size, and body label size.
- Support richer annotations, bigger captions, and alternate canvas sizes for PPT reuse.
- Keep outputs readable after insertion into DOCX and after screenshot/cropping into slides.

### 3. Literature management

Current status:

- The thesis body has a reference list, but there is no dedicated tool that tracks Chinese-vs-English balance or validates whether each added citation is reflected in the正文 discussion.
- The current reference section is overwhelmingly English-heavy.

Upgrade direction:

- Use `scripts/audit_reference_balance.py` to measure the mix.
- Add more Chinese references to the literature review, theory chapter, and domestic research status discussion.
- If the project keeps growing, add a structured bibliography source later, such as JSON, YAML, or BibTeX.

### 4. PPT preparation toolchain

Current status:

- There is no project-local PPT extraction or generation tool yet.

Upgrade direction:

- Keep thesis sections and figures slide-friendly now.
- Later add a PPT-specific tool or asset set after the DOCX/export pipeline stabilizes.

### 5. Experiment scaling and throughput

Current status:

- The project has a stable larger `FLEURS` baseline and a verified `run_aug_fleurs240` result.
- `download_commonlanguage.py` now works again, but the verified `Chinese_China + English + Japanese` export was smaller than the current `FLEURS` expansion, so it is not the preferred route for "10k+ training set" scaling.
- `download_voxlingua_subset.py` can export `zh/en/ja` subsets from `VoxLingua107`, which is the more promising larger-source candidate.
- However, a first medium `VoxLingua107` trial showed a practical bottleneck: raw audio decoding and CPU-side preprocessing can make initial multi-thousand-sample trial runs too slow.

Upgrade direction:

- Keep `VoxLingua107` as the likely next scaling dataset.
- Add a staged workflow: micro subset, medium subset, then formal run.
- Continue reducing CPU-side bottlenecks before committing to a full larger-dataset training campaign.
- Record runtime settings such as `device`, `num_workers`, and `pin_memory` in summaries so future diagnosis is faster.

## Thesis and experiment facts to preserve

- Main model line: `CNN + BiLSTM + Attention`.
- Stable expanded baseline: `FLEURS` three-language setup with `2160` total items and the verified best run in `run_aug_fleurs240`.
- Verified `CommonLanguage` three-language export summary:
  - train: `401 / 414 / 490`
  - validation: `98 / 79 / 122`
  - test: `100 / 98 / 144`
  - conclusion: smaller than the current expanded `FLEURS` baseline
- Verified `VoxLingua107` trial export summary:
  - `en`, `ja`, `zh`
  - `3` shards per language
  - `1500` wav files per language
  - useful for scaling trials, but not yet a stable official thesis result
- Existing figure filenames already map to chapter references in the markdown. Prefer regenerating them instead of renaming them casually.
