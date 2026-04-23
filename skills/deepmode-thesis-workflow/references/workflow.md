# Workflow

## 1. Build context from local materials first

Read these files before changing the thesis:

1. `../../论文材料/0-排版与提交规范清单.md`
2. `../../论文材料/1-毕业论文任务书_草稿.md`
3. `../../论文材料/3-文献综述_草稿.md`
4. `../../论文材料/12-毕业论文正文_基于深度学习的语言识别系统_文献修订版.md`
5. `../../language_recognition_system/README.md`
6. `../../language_recognition_system/checkpoints/run1/train_summary.json`

Use local evidence first. Treat external references as enrichment, not as a substitute for project reality.

Before claiming that the thesis is "done", also read [structure-content-checklist.md](structure-content-checklist.md).

## 2. Run the thesis iteration loop

1. Identify the active goal:
   - expand content
   - enrich literature
   - improve figures
   - repair DOCX formatting
   - prepare for PPT conversion
2. Audit the current bottleneck:
   - reference imbalance
   - weak chapter depth
   - unclear diagrams
   - fragile exporter
   - missing slide-ready summaries
4. Run `scripts/audit_thesis_structure.py` before major edits and after major edits.
3. Change the source of truth first:
   - thesis markdown for content
   - `generate_paper_assets.py` for figure logic
   - `export_md_to_docx.py` for repeated formatting problems
5. Regenerate affected outputs only after the content and structure are sound.
6. Summarize what changed and what still needs work.

For experiment-heavy turns:

7. Verify the scaling path before launching long jobs:
   - confirm the new dataset is actually larger than the current baseline
   - confirm per-language counts after export, not only from dataset cards
   - run a short or reduced-subset GPU trial first
8. If the first large trial stalls, times out, or is interrupted:
   - check once for orphaned worker processes
   - stop only the matching leftover training batch
   - downshift to a smaller trial instead of retrying the same command repeatedly

## 3. Expand literature the right way

Targets:

- Keep a substantial Chinese literature share. Default target: at least one third Chinese references.
- Do not add references only to pad the count. Every new citation should support a concrete claim or comparison in the thesis body.
- Balance source types: English conference/journal papers plus Chinese journals, theses, reviews, and application-oriented papers.
- New references may be used to deepen explanations of components that the project already uses. This is often the safest and most thesis-appropriate way to expand content.

Preferred Chinese literature slots:

- 国内外研究现状中的国内研究部分
- 语音处理、语言识别、说话人识别相关综述
- 深度学习语音分类、端到端识别、自监督学习在中文语境下的综述或应用论文
- 中文硕博士论文，用于补足工程实现、系统设计、实验组织等 thesis-style discussion

Good expansion targets in this project:

- CNN and local time-frequency feature extraction
- BiLSTM and sequential context modeling
- attention pooling and utterance-level aggregation
- log-Mel feature extraction and preprocessing
- language identification evaluation metrics
- multilingual dataset construction and split strategy

Current experimental scaling notes for this project:

- The expanded `FLEURS` setup remains the stable verified baseline for the thesis.
- A three-language `CommonLanguage` export was verified locally and turned out smaller than the current `FLEURS` expansion, so it should not be treated as the default "bigger dataset" path.
- `VoxLingua107` is a better scaling candidate, but its raw audio decoding burden is much heavier; first-pass multi-thousand-sample trials should be treated as diagnostic runs, not immediately as the new official experiment.

Before and after literature changes:

- Run `scripts/audit_reference_balance.py`.
- Check whether the正文 discussion has actually absorbed the new references.
- Keep bibliographic details complete enough for final formatting.

## 4. Improve figures for both thesis and PPT

Always prefer editing `../../论文材料/tools/generate_paper_assets.py` instead of manually touching only exported PNGs.

Figure quality rules:

- Make labels large and readable.
- Make arrows readable after DOCX scaling and later PPT cropping, but never let them cover labels or enter the next node box.
- Use clear stage boundaries and avoid decorative clutter.
- Keep terminology identical to the thesis body and the codebase.
- Prefer diagrams that still work when cropped into one slide.
- Remove figure-internal numbering when the numbered caption below the image already exists in the thesis body.

If a figure becomes crowded:

- Split it into two figures instead of shrinking all labels.
- Or increase canvas size and export resolution.

## 5. Repair DOCX output systematically

Use `../../论文材料/tools/export_md_to_docx.py` as the normal export entrypoint.

Improve the exporter when you need:

- better heading styles
- better figure and table captions
- better title and abstract formatting
- better appendix layout
- better bibliography handling
- school-template alignment
- top-and-bottom wrapped centered images at near-maximum safe width
- real equation objects with equation numbering instead of raw markdown or LaTeX source

Avoid repeated manual patching of generated `.docx` files unless the user explicitly wants a one-off final polish.

Do not use exporter improvements as a substitute for repairing weak argument flow, missing references, or incomplete chapter logic.

## 5.5 Diagnose slow training before scaling further

If the user reports "CPU full, GPU only around 20%":

- Verify whether the checkpoint `config.device` is `cuda`.
- Verify `torch.cuda.is_available()` on the current machine.
- Inspect the input pipeline before blaming the model:
  - `num_workers`
  - `pin_memory`
  - whether waveform augmentation is CPU-side
  - whether feature extraction is recomputing windows/filterbanks every sample
- In this project, cached log-Mel components and parallel workers should be preferred before assuming the model itself is too small.

## 6. Prepare the thesis for later slide conversion

While editing the thesis:

- End dense subsections with a short takeaway sentence.
- Prefer one main idea per subsection.
- Add comparison tables where a future slide will need quick contrast.
- Keep system architecture, training flow, deployment flow, and experiment result figures clean and self-contained.

Slide-friendly chapter rhythm for this project:

1. Research background and motivation
2. Related work and literature comparison
3. Proposed model and data flow
4. Experiment setup and results
5. System implementation and deployment
6. Conclusion and future work

## 7. Expand existing structures instead of inventing new ones

When a chapter feels thin, first ask which real project components are already present but under-explained.

Preferred move:

1. Find the real component in the code or current thesis.
2. Add a short theoretical explanation.
3. Add literature support.
4. Explain why that component is appropriate for this project.
5. State any tradeoffs briefly.

If visible word count in WPS/Word is still too low after a markdown expansion:

1. Measure visible thesis text, not only raw markdown length.
2. Expand weak explanatory sections in Chapters 2 to 5 first.
3. Prefer deeper mechanism explanation, experiment interpretation, engineering rationale, and system robustness discussion over filler text.

Examples for this project:

- Expand `CNN` from a one-line mention into local receptive field, parameter sharing, and time-frequency pattern extraction.
- Expand `BiLSTM` into forward/backward context modeling and why bidirectional encoding helps language discrimination.
- Expand attention pooling into weighted aggregation, key-frame emphasis, and why it is stronger than simple averaging in many speech tasks.

## 8. Current runbook note

As of the current project state:

- `download_commonlanguage.py` has been updated to use the parquet-conversion branch because the legacy dataset script path no longer works with the installed `datasets` stack.
- `download_voxlingua_subset.py` is available for shard-limited `VoxLingua107` exports.
- Direct long training on the first medium `VoxLingua107` subset is currently too slow to be the default next step; prefer micro-subset validation first, then scale.
