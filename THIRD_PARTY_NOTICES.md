# Third-party sources and licenses

## LatentMAS

The historical runtime includes a pinned subset of upstream LatentMAS under `snapshot/bundle/runtime/vendor/LatentMAS/`. Its original Apache License 2.0 is preserved at that directory's `LICENSE`; original source files and README are unchanged. Treat this as upstream work, not as a new contribution of this repository. Per-file SHA256 is recorded in the publication manifest and original bundle lock.

## HeadOBF

Upstream: https://github.com/markli404/When-Less-Latent-Leads-to-Better-Relay
Pinned revision: `36f43183e84d2fbf3b84c9bf4e3e06b7c442a523`.
The HOBF and BaseKVCompressor files are NOT redistributed in this snapshot. Their URLs and expected SHA256 remain in `snapshot/run/author_manifest.json`. No LICENSE file was found at that revision in the checked repository tree. The optional fetching helper does not grant a license; obtain appropriate rights for your intended use.

## Model and datasets

Qwen3-8B model files and GPQA question text/options/gold labels are not included. Obtain them separately under their applicable licenses/access terms. A model SHA lock and question-ID/hash manifest are provided for provenance only.

## This project's original code

No repository-wide redistribution license has been selected for the original project code in this initial snapshot. A public repository alone does not grant a general redistribution license. Third-party files retain their own terms.
