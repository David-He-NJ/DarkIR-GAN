# Stage III DarkIR-GAN

This directory contains the matched adversarial branch used for the paper. It
uses `training/baseline_continued/` as the authoritative generator recipe and
imports its generator, dataloader, reconstruction losses, optimizer factory,
scheduler, validation and metric logging directly.

## Controlled comparison

```text
baseline_continued generator objective
vs.
baseline_continued generator objective + 0.01 * PatchGAN LSGAN loss
```

The generator-side settings are guarded against the continued-baseline YAML.
A mismatch aborts before training. Both branches restore generator weights,
optimizer state and scheduler state from the shared Stage II checkpoint.

The retained reconstruction objective and the added adversarial term are:

| Component | Effective coefficient / behavior |
|---|---|
| Pixel L1 | `1` |
| LPIPS object | `0.01` |
| Laplacian edge object | `25` |
| Baseline enhancement object | `0.25`, evaluated on `(output, target)` |
| Adversarial | LSGAN MSE, outer weight `0.01` |

The discriminator is a five-layer PatchGAN trained with real label `0.9`, fake
label `0.0`, and Adam using learning rate `1e-4`, betas `(0.5, 0.999)`, and
cosine annealing to `1e-6`.

## Initialization

The run requires:

```text
training/baseline/experiments/checkpoints/baseline_best_psnr.pt
```

A missing or incomplete checkpoint is a hard error. The loader restores and
verifies the generator, optimizer and scheduler states, checks `intro.weight`,
computes the checkpoint SHA-256, and records configuration/source hashes and
restored optimizer metadata in `experiments/run_manifest.json`.

## Run

After preparing the documented environment, dataset, IQA weights and shared
initialization:

```bash
python training/gan/verify_setup.py
(cd training/gan && python smoke_test_controlled.py \
  --iterations 100 --validation-batches 2 --run-name smoke_strict)
bash scripts/run_stage3.sh gan
```

`verify_setup.py` is a standard-library-only static guard. The smoke test is a
bounded GPU run; it does not enter the full 30-epoch loop.

## Verified run

Two 100-iteration smoke runs completed successfully on AutoDL. The stricter run
checked finite, non-zero generator and discriminator gradients, optimizer and
scheduler restoration, validation metrics, checkpoint round-tripping and
stable post-warmup GPU memory.

The formal run completed 30 epochs with exit code 0 and ten full validations on
all 1,800 validation images. Its final checkpoint metrics were PSNR 24.9401,
SSIM 0.8553, LPIPS 0.1301, MUSIQ 60.7176, NRQM 5.9831 and NIQE 4.7371. Raw
records and provenance hashes are under `../../results/stage3/`.

The public directory name is `training/gan/`; the completed AutoDL deployment
used `training/gan_controlled/`. The verified production script, configuration
and smoke-test file contents are unchanged by this directory relocation.

This evidence establishes runtime execution in the recorded environment. It
does not remove the single-run and non-batch-identical limitations documented
in `../../docs/reproduction.md`.
