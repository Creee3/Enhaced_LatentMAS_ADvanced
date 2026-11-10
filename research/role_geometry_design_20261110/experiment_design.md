# Experiment Design: Role-Source Geometry and Relay Interference

## Goal

Find out whether role-origin KV has measurable structure, whether global readout error hides source-block distortion, and whether true role boundaries—not arbitrary chunk boundaries—help preserve downstream receiver behavior. Work in three gates; do not add an adaptive algorithm before the mechanism survives.

## Phase 0 — Offline archive reanalysis (completed)

The frozen 576-question archive includes per handoff, layer and KV head `total_mse` and `true_blocks_mse`, but no raw role K/V or vector-valued block errors. Four blocks include protected sink/self plus Planner, Critic and Refiner provenance.

Define a descriptive cancellation index:

`C = 1 - mean(total_mse) / (4 * mean(true_blocks_mse))`

Positive C means the squared total error is lower than the energy in the four block errors, consistent with cross-term cancellation. It is not a direct measure of information or answer quality. Since T optimizes total error by construction, higher C under T is expected/endogenous and cannot establish that T damages downstream behavior.

Mean C across 576 questions:

| Handoff | GGG | RRR | TTT |
|---|---:|---:|---:|
| 1 | .272 | .313 | .400 |
| 2 | .174 | .126 | .366 |
| 3 | .194 | .147 | .369 |

TTT's C is highest at each handoff. This is endogenous to T's total-error objective and is not evidence of a defect. RRR's C is higher than G at handoff1, lower at handoffs2-3. At handoff1, block0 also contains the fixed task prefix before the first source segment, so its C cannot be interpreted as role-vs-role cancellation. Later handoffs are associations because each method reaches a different cache state. Existing logs do not identify which role pair cancels or support effective-rank/subspace analysis.

## Phase 1 — Compact geometry instrumentation on a shared full-cache trajectory

Use one frozen full-cache trajectory per question. At each receiving prompt, use the already-required receiver probe and current cached K/V; no extra forward per token. Stream compact metrics to private output, then discard tensors. Do not alter the active 705-question source/run.

Collect by question × handoff × source × token class (prompt versus generated latent) × layer × KV head:

- K and V centered covariance participation ratio `(tr C)^2 / tr(C^2)`;
- entropy effective rank from normalized singular-value energy;
- fixed-q normalized top-q singular spectrum;
- equal-rank pairwise subspace overlap `||U_r^T U_s||_F^2/q` for K and V;
- source-specific full-vs-selected readout error vector norm and cosine between source error vectors;
- existing aggregate cancellation C.

Post-RoPE K is the operational cache geometry. If the cache adapter exposes pre-RoPE K without additional forwards, report it separately to distinguish content geometry from position rotation. Store scalar summaries and provenance hashes only; do not retain raw K/V, question text, answer text or generated tokens.

### Matched nulls

- Equalize counts by repeated fixed-seed subsampling.
- Stratify/condition on relative-position-age bins and prompt/generated-latent token type.
- Within each role, create pseudo-source chunks matched in size to actual roles. This tests generic chunking.
- Treat question as the independent bootstrap unit; layer/head are repeated measurements, not independent N.
- Report each dataset and handoff separately; Diamond is a GPQA subset, not an added independent sample.

H1 support requires role-source geometry to exceed matched pseudo-source variability and remain after position/token-type control. Otherwise conclude the role label alone is not a useful geometry descriptor.

## Phase 2 — Same-state boundary intervention

From the exact same pre-compression cache/probe tensors, replay selectors offline with identical candidate bank, budget, protection, RNG and original K/V/RoPE gather:

1. G attention-mass selection.
2. T total readout-distortion selection.
3. R true-source block distortion.
4. R-shuffle using multiple pre-registered permutations that preserve per-source block sizes and prompt-position strata.
5. Position-matched pseudo-source R as an arbitrary-chunk null.

First compare only selector metrics on the same snapshots. Next fork the receiving model from the same prefix/RNG and measure next-token KL to Full-cache logits, receiver hidden-state divergence, and—on a held-out task set—paired answer correctness. Repeat from a common pre-compression prefix at each handoff so downstream comparisons are paired; do not treat caches from separately compressed earlier branches as identical states.

Primary mechanistic outcome: source-block readout distortion and cancellation index.

Primary behavioral outcome: next-receiver KL to Full cache on a fixed pre-registered set of decode positions.

Secondary endpoint: end-task paired exact correctness and parse validity.

H3 support requires true-boundary R to beat both size/position-preserving shuffled labels and arbitrary pseudo-source blocks at the same bytes, and the gain must carry to receiver fidelity. If only a training-set diagnostic changes but next-agent behavior does not, the proposed local objective is not sufficient.

## Dataset Selection Gate (must be locked before any new experiment)

Candidate minimum diagnostic pilot: 20 questions each from AIME24, GPQA-Diamond, GPQA-nonDiamond, HumanEval+, MedQA and MMLU-Pro (120 unique questions total), using the same selected IDs for all selector arms. This spans olympiad math, expert science, coding, medical QA and broad knowledge. GSM8K is excluded because it is near ceiling in the current results.

Use deterministic ranking `SHA256(salt || dataset_group || stable_question_id)` over the frozen manifests, with selection made without looking at answer correctness or method outcomes. Select 20 by ascending hash for AIME24, GPQA-Diamond, GPQA-nonDiamond, HumanEval+ and MedQA. For MMLU-Pro, select one lowest-hash row from each category, then six additional globally lowest-hash rows. Keep a manifest of IDs, source index, category/stratum, hash and salt; Diamond is reported separately and is not added as an independent GPQA dataset.

All current benchmark records have been viewed, so this 120-question set is exploratory regardless of deterministic SHA selection. Twenty questions per group is adequate for checking whether the streamed geometry/readout metrics work and for estimating variance; it is not adequate to assert task-accuracy superiority, equivalence or broad generalization. A new held-out task or dataset is needed for confirmation.

## Phase 3 — Small execution sequence

1. CPU fixture: synthetic tensor tests for metrics, rank, overlap and label permutations; no model forward.
2. Engineering diagnostic: 6 questions (one per selected group) from the locked ID manifest, fixed Full-cache trajectory, all statistics streamed and discarded. Check shapes, non-finite values, protection, same-state replay and aggregate-vs-direct calculations; do not score/select by correctness.
3. Exploratory mechanism pilot: all twelve methods Full, Gen, H, L, H-OBF, L-OBF, G, T, R, R-shuffle, Pseudo-R and Random-matched run complete three-handoff trajectories on 120 questions. G/T/R repeat at every hop, corresponding to GGG/TTT/RRR. Per-hop diagnostics are attached measurements; Full trajectory snapshots additionally permit paired same-state selector and receiver comparisons. Engineering keys are reused. Total 1440 keys in four scheduling batches of30. Report all endpoints independently by dataset group.
4. Confirmatory: choose a new held-out task/question set before examining pilot effects. Freeze one chain-level primary comparison (RRR vs H-OBF/L-OBF for method novelty, or RRR vs RRR-shuffle for boundary specificity), one continuous fidelity endpoint and one answer endpoint. Treat local R vs R-shuffle as a mechanism analysis and GGG as an internal global-attention control. Run paired two-sided exact McNemar separately within each dataset group for binary correctness; bootstrap by question separately within each group for continuous outcomes. Declare any cross-dataset family correction in advance. Size the held-out set from per-dataset pilot variance/effect estimates rather than declaring 20 per group sufficient.

If generation randomness is material, use a small fixed list of seeds per question; question remains the cluster and seed is a repeated measure.

## Follow-on method only if H1-H3 survive

At a fixed total prompt-KV budget, allocate positions by each source's marginal downstream distortion rather than giving every source an implicit equal criterion. Add a subspace-redundancy penalty only if measured overlap is stable and predicts redundant contributions. Compare fixed R, adaptive role budget, adaptive budget plus geometry penalty, G/T and R-shuffle. This keeps added complexity justified by observed structure.

## Limits

- Existing fixed datasets are already seen; same-task follow-ups remain exploratory.
- Roles are entangled with templates, content and position.
- Local attention outputs do not equal final logits or task answers.
- Full-vs-R/G comparisons across the two completed runs are exploratory.
- KV byte ratios do not establish inference speedup.
- No new run is launched by this design document.

## Scope Correction — 2026-11-10

The authoritative execution count is **120 questions × 12 methods = 1440 method-question keys**. The 12 methods are Full, Gen, H, L, H-OBF, L-OBF, G, T, R, R-shuffle, Pseudo-R and Random-matched. Each method runs a full three-handoff trajectory; per-hop geometry and receiver measurements are attached diagnostics, not extra answer keys. Queue scheduling uses four batches of 30 questions; 30 is a batch size and does not mean 30 questions per dataset.
