# Dataset Selection Gate — Role Geometry Pilot

## Status

User-selected scope recorded and SHA manifest generated. No new experiment is running from this design. The execution protocol and launch remain pending; do not start while the unrelated active cloud run owns GPU0/1.

## Locked Candidate Scope

Use 20 questions from each of six LatentMAS benchmark groups (120 unique questions total): AIME24, GPQA-Diamond, GPQA-nonDiamond, HumanEval+, MedQA, and MMLU-Pro. GSM8K is intentionally excluded because the current run is near ceiling and the user considers it too weak for this mechanism pilot. Use the same IDs across local G/T/R/R-shuffle diagnostics and chain-level GGG/RRR/TTT (with optional RRR-shuffle boundary null). A six-question engineering check, one per group, comes first and validates only mechanics, not task accuracy.

Why this scope: it covers olympiad math, expert science QA, code generation, medical QA and broad knowledge while keeping Diamond/non-Diamond as separate GPQA strata. MMLU-Pro uses 20 globally but is category-stratified so the pilot is not concentrated in a few subjects.

## Deterministic Sample Rule

Before extracting the sample, lock the manifest revision/SHA and a new salt. Rank each item by `SHA256(salt || dataset || stable_question_id)` ascending. Do not inspect answer labels or model performance while choosing.

- AIME24: first 20 of 30 by ascending hash.
- GPQA-Diamond: first 20 of the 198 Diamond items by ascending hash.
- GPQA-nonDiamond: first 20 of the 250 non-Diamond items by ascending hash.
- HumanEval+: first 20 of 164 tasks by ascending hash.
- MedQA: first 20 of the 300 official LatentMAS JSON rows by ascending hash.
- MMLU-Pro: select one lowest-hash item from each category, then six additional globally lowest-hash items; total20.

Persist only IDs, stratum/category, manifest reference, salt and ranking hash in a private selection manifest. Diamond remains a GPQA subset, not an extra dataset to add to GPQA N.

## Interpretation Boundary

N=20 per group is suitable for testing diagnostic instrumentation, checking direction/variance of continuous geometry metrics, and deciding whether a larger study is worthwhile. It is not a basis for accuracy superiority, equivalence or confirmatory generalization claims. Layers/heads/query positions are repeated measures; question is the resampling unit. All current datasets have been viewed, so this pilot is exploratory. A new held-out task or dataset is needed for confirmation.

All six groups are analyzed independently. No pooled accuracy, pooled primary effect, or pooled significance test is planned for this pilot. Any overall visual summary comes after the six per-group reports and is descriptive only.

## Local Precedent

- Our 2026-11-08 G/R/T run used four engineering questions (two GPQA stratified by Diamond status, one MMLU-Pro, one GSM8K): 12 answer keys. Its purpose was structural/billing/archive gating, not accuracy inference.
- Our 2026-11-02 selector smoke test used two GPQA questions (one Diamond, one non-Diamond), repeated three cache reconstructions and rotated T/R/R-shuffle order. Repeats were for instability detection, not six independent samples.
- CacheBack's paper used 13 specially eligible FanOutQA development questions for evidence-recall selector diagnostics and parameter choice, then separate end-to-end benchmark evaluation. That is evidence a small diagnostic development set can validate a mechanism proxy, not evidence that N=13 or20 supports end-to-end efficacy.
- OBF's paper evaluated a nine-benchmark suite. It selected H/L rank settings on Qwen3-4B before testing larger backbones and included component/design ablations; it did not establish a universal 20-question validation recipe.
- LatentMAS evaluated nine benchmark datasets and reports three independent runs for its configured experiments; its scale is full benchmark evaluation, not an engineering-smoke precedent.

## Decision Needed

The six-group scope is recorded from the user's explicit choice. The private selection manifest is generated and verified with 120 unique stable IDs. Only the execution protocol and launch remain; do not start while the unrelated active cloud run still owns GPU0/1.
