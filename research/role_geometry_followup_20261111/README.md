# 120-question role-conditioned KV follow-up

This directory contains the 2026-11-11 diagnostic experiment source and protocol. It studies whether changing the assigned task (planning, checking, refining) changes KV geometry under a shared history and matched question-token positions. It does not generate Judger answers or evaluate RRR accuracy.

## Scope

- Six dataset groups, 20 previously selected questions each; 120 questions total.
- Three fixed history conditions, three task roles, two phrasings per role.
- Prompt/shared-question and latent samples measured separately; K and V measured separately.
- Three native Full reconstruction calls plus 18 intervention calls per question, 2,520 unique request IDs total.
- At most six resident workers, three each on GPU0 and GPU1, with an engineering audit/archive gate before formal work.

`protocol.md` describes the controls and limits. `measure.py` implements centered spectral metrics and top-four subspace comparisons. `worker.py` collects the measurements; `audit.py`, `collect.py`, and `verify_live.py` check evidence and accounting. `interpretation_audit_20261111.md` corrects the external interpretation, including the positive-question count of 115/120 for ctx1 prompt K.

## Safe local mechanics check

Requires Python and NumPy:

```sh
python -m unittest discover -s research/role_geometry_followup_20261111 -p test_mechanics.py -v
```

This check makes no model calls and launches no cloud processes.

## Runtime prerequisites

This is an experiment source release, not a standalone model/runtime bundle. Cloud execution expects the exact archived parent runs, frozen runtime, model lock, dataset manifest, and shared budget ledger referenced by `common.py` and the preparation scripts. Those dependencies must be restored and hash-verified before execution. Private evidence paths in the scripts describe the original environment; this branch does not include their contents.

Question text, labels, raw answers, request logs, sampled KV tensors, model weights, credentials, and shared budget ledgers are excluded. The original scientific files are copied without editing their frozen contents. Historical notes saying “local Git, no push” describe the earlier authorization; the user explicitly authorized GitHub publication on 2026-11-11.

## Status at publication

All 120 diagnostic question records exist and the formal structural audit passed. The local controller then exited after a 600-second SSH timeout during collection/archive processing. Full formal archive verification and the gated final summarizer have not completed. Exploratory read-only aggregates are not a substitute for those archive gates; no gate was relaxed for this release.

This branch publishes this directory only, based on the existing GitHub main branch. It does not publish the other unpushed local experiment history or unrelated working files.
