# Research Question Card: Role-Origin KV Geometry

## Question

Do KV segments grouped by source role have measurable geometry or readout contributions that a global token score misses, and does retaining source-conditioned contributions improve the next receiver's behavior at a fixed cache budget?

## Motivation

The supervisor's premise is that different roles may produce KV with different mathematical properties. Current RRR labels three source blocks and sums their readout distortion, but does not model role-conditioned rank, K/V subspace overlap, or inter-source redundancy. Test the premise before adding complexity.

## Existing Evidence

- GPQA96: Full47, GGG57, RRR56. GGG is +10 vs Full, paired exact p=.0129; RRR is +9 vs Full, p=.093; RRR is -1 vs GGG, p=1.000. These are exploratory cross-run comparisons.
- MMLU-Pro280: Full199, GGG191, RRR198. RRR is -1 vs Full, p=1.000; +7 vs GGG, p=.189.
- GSM8K200: Full197, GGG197, RRR196; differences are one item.
- Same-task comparisons are exploratory because these questions were already examined; Full vs GGG/RRR also comes from a separate completed run.
- Archived local readout diagnostics suggest block-error cancellation: TTT has larger cancellation than RRR at all three handoffs; RRR is lower than GGG only on handoffs 2 and 3. This proxy does not establish answer-level causality.

## Hypotheses

H1 Geometry: At matched layer, KV head, token type, count and relative-position distribution, role-source K/V have effective-rank or subspace-overlap patterns beyond within-role pseudo-source partitions.

H2 Interference: On the same pre-compression receiver snapshot, a total-readout objective can hide large source-block errors through cancellation; the cancellation measure predicts the next receiver's divergence from the Full-cache reference.

H3 Boundary specificity: At equal KV bytes, true-source R reduces source-block error and next-receiver divergence more than R with size-preserving shuffled source labels.

## Falsifiers

- H1 fails if between-role differences sit inside position/token-matched pseudo-source variability or vanish under template controls.
- H2 fails if cancellation does not predict next-receiver divergence, or lower blockwise loss does not improve receiver fidelity.
- H3 fails if shuffled labels match true labels within a predeclared practical margin, or if true R does not improve over G/T on the same snapshot.
- The broad role-geometry story fails if no repeatable source-conditioned geometry appears and the boundary intervention has no effect.

## Evidence Status

Exploratory design on previously viewed fixed datasets. Any confirmatory claim needs a new held-out task/question set and pre-registered paired protocol.
