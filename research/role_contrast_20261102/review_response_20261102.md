# Pre-generation review: source boundaries, scaling and diagnostics

2026-11-02. Review of the user's pasted critique. No model calls, cloud launch or parameter search. The lambda=1 implementation remains an initial mechanical prototype, not a generation-ready frozen method. This review supersedes treating its initial scalar choice as an experimental commitment.

## Accepted

- The contrast loss is a block/role matching penalty, not a new decomposition theorem. A real-boundary versus size/structure-matched pseudo-boundary control is needed.
- First handoff has one producer, so centered role contrasts vanish; with the same candidates and tie rules, T and R coincide there.
- Protecting all latent limits the claim to prompt selection and associated readout preservation. It does not establish selective compression of latent reasoning contributions.
- Use prompt queries for selection, queries unused by ALL selection steps for diagnosis, preserve GQA query-head distinctions, report role attention mass, and predeclare the main paired comparison.
- Backward deletion can cache per-query, per-role numerators and selected mass instead of rebuilding attention every trial. End-to-end cost remains to be measured.
- Do local/mechanics and representational diagnostics before new terminal-answer generation.

## Mathematical qualifications

For E_r=C_r-C_r,S and zero self/shared output error:

    L = R^2||E_mean||^2 + lambda/R * sum_r||E_r-E_mean||^2.

At lambda=R^2 this equals R*sum_r||E_r||^2, so optimizers coincide with role-block matching. The factor R is not literal equality with the unscaled sum. On the orthonormal mean coordinate sqrt(R)*E_mean and centered-error subspace, coefficient ratio is R^2/lambda, not the apparent27 obtained from differently scaled objects when R=3. More importantly, lambda=1 does not imply the role term has negligible numerical influence: if mean errors cancel, it may be the only nonzero term. Check actual term sizes, overlap and held-out distortion; do not infer a zero effect from coefficients alone.

The self/shared KV is fixed, but its OUTPUT is generally not fixed: eviction changes the softmax denominator. Therefore the zero-self-error simplification is not the actual general implementation. A cleaner next formulation to evaluate is explicit block matching including block0=self/shared, without claiming that a scalar substitution gives exact equivalence under real masks.

Protected latent values also have changing readouts. With full normalized weights a and retained mass z_S, C_latent,S=C_latent/z_S for fixed queries and retained latent rows. Thus contrasts do not change ONLY through prompt numerators. A prompt-only action space can alter latent effects through renormalization and later query drift. This may merely enforce attention-mass preservation, not protect semantic novelty.

Decomposing contrast energies into prompt/latent fractions requires the cross term:

    ||D||^2=||D_prompt||^2+||D_latent||^2+2<D_prompt,D_latent>.

Likewise include cross terms for question/instruction components. Report changes after eviction and not just uncompressed component energy. Do not assume earlier roles always receive more attention.

The exact cancellation counterexample in the CPU test is valid: remaining numerator and total output are both zero, so renormalization preserves zero. It is a special case. Generally for removed mass zeta and removed numerator X:

    Y_new=(Y-X)/(1-zeta).

Removing a zero-sum pair does not preserve a nonzero Y. Keep the special example explicitly qualified and add a nonzero shared-output case before using the intuition broadly.

## Actual cache layout

Existing stage_ops uses indices[layers,KV_heads,new_len]. Layers/heads may retain DIFFERENT original positions, but every head retains the SAME number. Tensor layout remains rectangular and KV bytes match. For a comparable extension, retain this per-head equal-quota layout and stack all query heads of each GQA group when evaluating its loss. No global cross-layer selection sum is necessary if each head is optimized separately. Layer-normalized loss is useful for diagnostics. A single shared row set across every layer/head is a different restriction and should not be silently introduced.

## Pseudo-role control

All physical candidate/protected sets, query tensors, budgets, bank, numerical rules and tie-breaking must be identical. Only the grouping labels change. A global role-name permutation leaves the symmetric loss invariant and is not a valid control. A predeclared shifted-boundary grouping should preserve sizes and be checked to mix true roles; document prompt/latent composition and positional structure. A shift coinciding with block lengths can accidentally be only a relabeling. One offset is exploratory; general boundary claims need stronger robustness. Lack of significant difference is not proof of equality.

## Data and evaluation prerequisites

Raw KV was NOT retained as a cross-question library. Existing archives hold answers, requests, positions and receipts, not the full per-query Q/K/V needed for this diagnostic. Regenerating intermediate states requires new metered model forwards even if no final answer text is generated. Define and distinguish an official Full trajectory from the local current available pre-eviction cache teacher.

For a controlled third-handoff diagnostic, all selectors must start from the SAME agreed producer trajectory/cache. Fixed-query diagnostics on that cache are off-policy for a recursively compressed new method, so they cannot replace actual new-method inference.

Original GPQA answers are free-form generations. First-token A/B/C/D probabilities are not automatically their answer-choice distribution. Any KL or correct-choice log-probability requires an explicit common scoring prefix and option-string tokenization/normalization, separate from the primary frozen-parser accuracy. Such logits were not saved and are not free to recover from answer texts.

main448 already CONTAINS Diamond198. If main448 is run at latent10, the Diamond subset has also been generated and must only be separately summarized, never generated again. Only latent40 can be deferred as a new setting.

## Recommended revised gate

First: fix role-block semantics (including shared block), headwise layout and fake boundaries; then measure per-role mass, prompt/latent/question/instruction response and cross terms, controllable errors, selection changes and unused-query distortion. Jaccard0.9 is a resource heuristic rather than a mathematical futility threshold; a few critical changed rows/heads can matter. If no useful controllable role signal or real-boundary advantage is found, stop this prompt-only candidate instead of scanning weights.

Do not start the old32-question answer pilot automatically. Any diagnostic forward pass needs a concrete metered plan; this review authorizes no cloud job.
