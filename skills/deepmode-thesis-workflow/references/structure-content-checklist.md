# Structure And Content Checklist

Use this checklist before declaring the thesis "ready" or "substantially complete".

## Non-negotiable principle

There is no local tool that can automatically generate a qualified graduation thesis. Existing scripts only:

- audit structure or reference balance
- regenerate figures
- export markdown to DOCX

The actual thesis quality still depends on careful editing, evidence-based writing, and consistency with the project code and experiment outputs.

## Required top-level structure

The thesis markdown should contain:

1. Title line with the thesis name
2. `## 摘要`
3. Chinese `关键词：`
4. `## Abstract`
5. English keywords line such as `Key words:`
6. `## 注释表`
7. `## 第1章`
8. `## 第2章`
9. `## 第3章`
10. `## 第4章`
11. `## 第5章`
12. `## 第6章`
13. `## 参考文献`
14. `## 附录`

## Content expectations by chapter

### Chapter 1

- Explain research background, meaning, domestic and foreign status, and the thesis contribution boundary.
- Avoid fake "original innovation" claims that are unsupported by the code or experiments.

### Chapter 2

- Explain task definition, preprocessing, features, model foundations, and evaluation metrics.
- Tie theory to the actual project implementation.
- Do not stop at naming modules. If the project uses a structure such as convolution, recurrent modeling, attention, or log-Mel features, explain the mechanism and why it matters for the thesis topic.

### Chapter 3

- Explain why the chosen model route is reasonable.
- Compare the route with literature rather than presenting it as arbitrary.
- Use real figures and experiment outputs.
- Expand existing model components into literature-backed subsections when needed. For example, if the model contains `CNN`, `BiLSTM`, and attention pooling, each can be explained more fully without pretending they are new inventions.

### Chapter 4

- Present optimization or extension thinking clearly.
- Keep it framed as an extension path if the codebase has not fully implemented it.

### Chapter 5

- Map the paper to real system modules, scripts, files, and deployment flow.
- Do not reduce this chapter to screenshots only.

### Chapter 6

- Separate completed work from future work.
- Make sure future work does not pretend to be already finished.

## Reference expectations

- Reference count should normally reach at least 20 for this project baseline.
- A substantial share should be Chinese references. Default target: at least one third.
- New references should appear in正文 discussion, not only at the end.

## Figures

- Every core figure should support the surrounding paragraph.
- Labels and arrows should remain readable after DOCX insertion.
- Figures should be reusable in future PPT conversion.

## Final sanity checks

- Terminology matches the codebase: `CNN + BiLSTM + Attention`, `FLEURS`, `train_summary.json`, `best_model.pt`.
- Performance numbers match `../../language_recognition_system/checkpoints/run1/train_summary.json`.
- The thesis claims do not exceed what the local project actually implements.
- The thesis does not leave major real components unexplained. Core structures already used by the project should have enough theory and literature support to stand on their own in the paper.
