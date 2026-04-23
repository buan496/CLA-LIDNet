---
name: deepmode-thesis-workflow
description: Project-specific workflow for the H:\deepmode graduation thesis on deep learning-based language recognition. Use when Codex needs to guarantee thesis structure and content quality, audit local thesis requirements, improve or expand the thesis markdown, add literature with a substantial share of Chinese references, refine thesis figures, reuse experiment outputs from language_recognition_system, repair DOCX export issues, or reorganize the paper so it can later be converted into a PPT deck. Do not use this skill as a promise of automatic thesis generation; use it for audit-first, content-first thesis iteration.
---

# Deepmode Thesis Workflow

Use this skill for work inside `H:\deepmode` that touches the graduation thesis, its references, its figures, or its export pipeline.

## Start Here

Read local requirements before editing content or scripts:

1. `../../论文材料/0-排版与提交规范清单.md`
2. `../../论文材料/1-毕业论文任务书_草稿.md`
3. `../../论文材料/3-文献综述_草稿.md`
4. `../../论文材料/12-毕业论文正文_基于深度学习的语言识别系统_文献修订版.md`
5. `../../language_recognition_system/README.md`
6. `../../language_recognition_system/checkpoints/run1/train_summary.json`

Then read:

- [references/project-tooling.md](references/project-tooling.md) before changing scripts or deciding what to upgrade.
- [references/workflow.md](references/workflow.md) before expanding literature, redrawing figures, repairing DOCX formatting, or preparing for PPT conversion.
- [references/structure-content-checklist.md](references/structure-content-checklist.md) before making claims that the thesis is complete.

## Use the Built-in Helpers

- Run `scripts/audit_thesis_structure.py` before and after major thesis revisions.
- Run `scripts/audit_reference_balance.py` to measure whether the reference list has enough Chinese literature.
- Run `scripts/rebuild_thesis_outputs.py --dry-run` to inspect the rebuild pipeline before making bulk output changes.
- Run `scripts/rebuild_thesis_outputs.py` after substantial thesis edits when figures or DOCX need to be regenerated.

## Follow These Rules

- Prefer local project requirements over imitation from external samples.
- Treat thesis structure and content as the primary deliverable. Exporters and figure scripts are only support tools.
- Do not describe scripts as if they can automatically write a qualified thesis. They can only audit, rebuild, or assist repeated tasks.
- Reuse real project code, metrics, file names, and experiment outputs. Do not invent experiments, datasets, or evaluation numbers.
- Add literature into the body discussion, not only into the reference list.
- Keep a substantial Chinese literature share. Default target: at least one third of the references should be Chinese unless the user asks for a different balance.
- Prioritize academically credible sources. For technical claims, prefer papers, theses, official benchmarks, journals, or conference proceedings.
- When the same formatting problem repeats, update the exporter or figure generator instead of applying one-off manual fixes to the generated files.
- Keep figures editable and slide-ready: large labels, clear layout, strong contrast, and enough whitespace for later PPT reuse.
- Figure-internal titles may stay, but figure-internal numbering should be removed when the thesis already has numbered captions below the image.
- Arrow size should improve readability without covering labels, nodes, or neighboring boxes. If thicker arrows cause overlap, enlarge spacing or shorten line endpoints instead of forcing bigger arrows through text.
- Preserve alignment between thesis wording and code reality. If the codebase says `CNN + BiLSTM + Attention`, do not silently rename it to a different architecture.
- When the user cares about final school submission quality, judge word count by visible thesis text as seen in WPS/Word, not only by raw markdown character totals.
- When expanding experiments with a new speech dataset, do not jump straight to a long full-scale run. First verify dataset size, per-language balance, clip duration burden, and a short GPU trial on a smaller subset.
- If a training run is interrupted or times out, check once for orphaned `conda/python` workers, stop that specific batch, and continue with a smaller diagnostic run instead of retrying the same long command in a loop.
- Low GPU utilization does not automatically mean the model is on CPU. In this project, confirm the saved checkpoint `config.device`, `torch.cuda.is_available()`, and whether CPU-side audio decoding, augmentation, and feature extraction are the real bottlenecks.

## Choose the Work Mode

### Tool Audit

- Read [references/project-tooling.md](references/project-tooling.md).
- Read [references/structure-content-checklist.md](references/structure-content-checklist.md).
- Confirm whether the task is about content, literature, figures, DOCX formatting, or PPT preparation.
- Upgrade the highest-priority bottleneck first instead of patching symptoms downstream.
- For experiment expansion, audit the data path before launching long training:
  - verify the dataset is actually larger than the current baseline
  - verify raw audio clips are not so heavy that first-pass trials become impractically slow
  - prefer a micro-subset or shard-limited trial before a formal run on the full new corpus

### Literature Expansion

- Use [references/workflow.md](references/workflow.md) and start from the current reference list in the thesis markdown.
- Run `scripts/audit_reference_balance.py` before and after literature changes.
- Prefer adding citations that support specific paragraphs, tables, model choices, datasets, or evaluation claims.
- Add Chinese references for domestic research status, engineering practice, speech processing reviews, and thesis-style background sections when the current list is too English-heavy.
- It is valid to expand explanations of structures that already exist in the project and thesis. If the system uses `CNN`, `BiLSTM`, attention pooling, `log-Mel` features, preprocessing, or standard evaluation metrics, deepen those sections with theory, mechanism explanation, advantages, limitations, and literature support.
- Prefer "existing structure + fuller explanation + literature support" over inventing new modules that the project does not actually use.

### Figure Refinement

- Edit `../../论文材料/tools/generate_paper_assets.py` before replacing exported PNGs by hand.
- Increase text size before adding more decorative detail.
- Use large but non-overlapping arrows. Prefer center-aligned diagrams, larger canvases, or split figures over arrows that block labels.
- Keep terminology synchronized with the thesis body and `language_recognition_system`.
- Favor figures that can be cropped or reused in slides without redrawing.

### DOCX and Formatting Repair

- Treat `../../论文材料/tools/export_md_to_docx.py` as the canonical export entrypoint.
- Update the exporter when you need better headings, captions, spacing, page structure, or template alignment.
- Prefer centered images with top-and-bottom wrapping unless local requirements clearly demand a different mode.
- For displayed formulas, export real equation objects and attach visible equation numbers instead of leaving markdown or LaTeX source code in the document.
- Rebuild after markdown edits instead of manually editing the DOCX unless the user explicitly asks for manual document surgery.
- Never let formatting work hide missing content or weak chapter logic. Fix content gaps first.

### PPT Preparation

- Write thesis sections so each subsection can become one slide-level idea.
- Prefer one strong figure or one comparison table per concept.
- Add short takeaway sentences after dense sections so slide extraction later is straightforward.
- Preserve chapter summaries, system architecture diagrams, experiment tables, and conclusion bullets because these convert well into PPT pages.

### Knowledge-Point Expansion

- When the thesis already mentions a real project component, expand it into a fuller academic explanation instead of leaving it as a short engineering label.
- Good expansion targets in this project include:
  - convolutional neural networks
  - recurrent and bidirectional recurrent networks
  - attention pooling
  - log-Mel spectrogram extraction
  - audio preprocessing
  - dataset design and split strategy
  - evaluation metrics such as Accuracy and Macro-F1
- Each expansion should answer three questions:
  - what the concept is
  - why it is suitable for this project
  - what limitations or tradeoffs it has
- Tie the explanation back to the local code and experiment pipeline. Do not write generic textbook content that floats free of the project.

### Experiment Scaling

- Reuse existing experiment evidence before promising a larger run.
- In this project, `FLEURS` with 240 items per split per language produced `2160` total items and a strong baseline; do not assume another dataset is larger until the exported counts are verified locally.
- `CommonLanguage` is available through a local downloader, but the verified three-language export was smaller than the current `FLEURS` setup and therefore is not the preferred path for scaling.
- `VoxLingua107` is the more promising larger-source option, but its raw audio is heavy enough that direct first-pass training on a multi-thousand-sample subset can still be slow.
- Preferred escalation path for new large datasets:
  1. downloader check
  2. per-language count verification
  3. micro-subset GPU trial
  4. medium subset trial
  5. formal run only after the above are stable

## Finish Well

- Report whether thesis structure and content checks pass.
- Regenerate only the outputs affected by the edits.
- Report which thesis files, scripts, figures, and outputs changed.
- Call out remaining gaps such as missing Chinese references, weak figure readability, or DOCX template mismatches.
