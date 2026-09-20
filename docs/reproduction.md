# Reproduction record

## Authoritative Stage III source

The completed AutoDL deployments identify GitHub commit `2f08ba5e73347d4e2e3862f81c156ff5089e00a1` as their source.

File-level SHA-256 comparison established:

- `training/baseline_continued/` is byte-identical between GitHub and the completed AutoDL baseline deployment.
- The public `training/gan/` directory is the paper's controlled branch. It
  relocates the deployed `training/gan_controlled/` directory while preserving
  the training configuration and optimization logic used for the reported run.
- The earlier pre-control GAN prototype is omitted because it changed multiple
  generator-side variables and was not used for the reported comparison.

No training recipe was reconstructed from prose after the run. The release
preserves the source that was actually deployed; `results/stage3/provenance.json`
records the relocated release paths and the original file hashes.

## Completed runs

| Branch | Status | Finished UTC | Full validations | Output images |
|---|---|---|---:|---:|
| Baseline-continued | exit code 0 | 2026-09-19 07:18:48 | 10 × 1,800 images | 80 |
| DarkIR-GAN | exit code 0 | 2026-09-18 06:46:43 | 10 × 1,800 images | 80 |

Both runs restored the same initialization:

```text
dbfdc6494daae9a809607f63e74a6193a8b4ed0004c02ae9f23dda1cfc723a99
```

The controlled manifest recorded generator optimizer learning rate `1e-6` and scheduler `last_epoch=30` after restoration. The discriminator was randomly initialized for the adversarial branch.

The full-validation CSVs and their audit hashes are retained under `results/stage3/`. Large checkpoints, TensorBoard events, dataset files and third-party metric weights are not committed. Checkpoints are distributed separately using the filenames and SHA-256 identities in [`model_assets.md`](model_assets.md).

## Before running

1. Install the recorded environment or an explicitly documented alternative.
2. Validate the dataset layout and the 20,400/3,600 train/validation image counts.
3. Place all four IQA metric files under `weights/iqa/`.
4. Place the shared Stage II initialization at the documented path and verify its SHA-256.
5. Run `python scripts/verify_release.py` and `python training/gan/verify_setup.py`.

## Remaining limitations

- The comparison is single-run. Repeated seeds, confidence intervals and paired per-image statistics are not available.
- The sampler updates its own paths after the dataset has captured initial path lists. Dataset, split, loader behavior and training length are matched, but batch order is not proven identical.
- The historical metric wrapper catches metric exceptions and can retain zero defaults. The completed CSVs are non-zero, but per-image failure counts were not logged.
- The IQA-PyTorch checkout used on AutoDL was installed from an unversioned local snapshot reporting `0.0.0`; its exact upstream commit was not preserved.
- The repository does not redistribute LOL-Blur or third-party IQA weights. Initialization and result checkpoints are separately hosted rather than committed to Git.

These boundaries should remain attached to any quantitative claim derived from this repository.
