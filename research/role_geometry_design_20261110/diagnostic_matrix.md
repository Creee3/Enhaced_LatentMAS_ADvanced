# Diagnostic Matrix: What the 120-Question Pilot Must Establish

## One 12-method run with per-hop diagnostics

Each of 12 methods produces one complete three-handoff trajectory and one final answer per question (120×12=1440 method-question keys). Per-hop geometry and selector diagnostics are fields attached to each trajectory, not an extra answer run.

## Local snapshot methods

Run every method on the same pre-compression KV/probe snapshot, with the same candidate bank, prompt budget, protected latent/sink rows, original K/V gather, RoPE positions and RNG:

| Method | Role in diagnosis |
|---|---|
| Full | No deletion reference |
| Gen | Generation/latent-only retention control |
| H | Per-layer/per-head attention selector |
| L | Per-layer shared-position attention selector |
| H-OBF | H-selected positions plus OBF value backfill; same H positions, modified retained V |
| L-OBF | L-selected positions plus OBF value backfill; same L positions, modified retained V |
| G | Global attention-mass control used by our GGG family |
| T | Total readout-distortion control |
| R | Source-block readout-distortion selector |
| R-shuffle | Same block sizes with source labels permuted |
| Pseudo-R | Same objective on position/age-matched pseudo-source blocks |
| Random-matched | Variance and candidate-count null |

H-OBF and L-OBF are not new position selectors: their position sets inherit H/L respectively, while the retained value states receive the paper's orthogonal residual backfill. Keeping them in the local matrix separates selection geometry from value-recovery effects.

Local methods should not be ranked by final task accuracy. Their purpose is to answer the mechanism questions:

1. **Geometry:** Do Planner, Critic and Refiner K/V have different effective rank or subspace overlap after matching age, token class and count?
2. **Interference:** Does global total error hide large source-block errors? Record the full cross-term matrix, not only a scalar cancellation index.
3. **Boundary specificity:** Does true R beat R-shuffle and Pseudo-R on the same snapshot?
4. **Predictiveness:** Do local source distortion and subspace metrics predict the next receiver's KL/hidden-state divergence?

## Full-chain interpretation

`Full, Gen, H, L, H-OBF, L-OBF, G, T, R, R-shuffle, Pseudo-R, Random-matched`

Every method is applied across all three handoffs, re-probing and continuing its own cache trajectory. G/T/R and their controls are repeated at each hop; thus G/R correspond to GGG/RRR chain variants. H-OBF/L-OBF inherit H/L position selection and add OBF V backfill. Queue batch size is 30 questions, four batches total; it is not a per-dataset sample count.

## Key content to extract

For each of the six dataset groups, each handoff, layer and KV head, retain compact summaries only:

- K/V participation ratio and entropy effective rank;
- top-q singular energy and pairwise K/V subspace overlap;
- per-source readout error vectors, norms and pairwise cosines;
- full cross-source error cross-term matrix;
- R versus R-shuffle and Pseudo-R deltas;
- next-receiver KL and hidden-state divergence from Full;
- actual KV bytes, selector time, receiver time and peak allocation.

Question is the independent unit. Layers, heads and decode positions are repeated measurements. Report each dataset independently; no pooled primary test.

## Decision tree

- If role geometry survives matched nulls, true R beats R-shuffle, and local metrics predict next-receiver divergence, the role-aware mechanism is supported.
- If geometry differs but R does not beat R-shuffle, role labels are descriptive but not a useful causal selector.
- If R beats R-shuffle locally but not at the next receiver, the local objective is insufficient for relay behavior.
- If RRR beats H/L/H-OBF/L-OBF at matched bytes and the gain is predicted by the local diagnostics, the chain-level method has a defensible contribution.
- If GGG and RRR are similar while RRR does not beat boundary controls, keep GGG as the simpler method and do not claim source-aware novelty.

## Scope Correction — 2026-11-10

The pilot is one 12-method full-chain run: Full, Gen, H, L, H-OBF, L-OBF, G, T, R, R-shuffle, Pseudo-R and Random-matched. This yields 1440 method-question keys for 120 questions. Every selector is re-applied at all three hops; local diagnostics are stored inside each method trajectory. Queue batches are 30 questions each, four batches total.
