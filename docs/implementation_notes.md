# Implementation notes

## Matched Stage III comparison

The reported comparison uses `training/baseline_continued/` as the
reconstruction-only control and `training/gan/` as the DarkIR-GAN branch. The
GAN branch imports the control's generator, dataloader, reconstruction losses,
generator optimizer and scheduler, validation routine, and metric logging.
Startup guards reject configuration drift before training.

Both branches use the same dataset split, 30-epoch continuation budget and
Stage II initialization. The checkpoint restores generator weights, optimizer
state and scheduler state. In the completed AutoDL runs, the restored generator
learning rate was `1e-6` and the scheduler state had `last_epoch=30`.

## Generator objective

The GAN branch retains the continued baseline's output clamp and effective
reconstruction objective. Loss classes apply YAML weights, while the training
loop applies the outer multipliers shown below.

| Component | Effective coefficient / behavior |
|---|---|
| Pixel L1 | `1` |
| LPIPS object | YAML `0.1` times outer `0.1` = `0.01` |
| Laplacian edge object | YAML `5` times outer `5` = `25` |
| Enhancement object | YAML `0.5` times outer `0.5` = `0.25`, evaluated on `(output, target)` |
| Adversarial | LSGAN MSE, outer weight `0.01` |

The adversarial treatment adds a five-layer PatchGAN discriminator, real-label
smoothing to `0.9`, fake label `0.0`, and a discriminator-only Adam optimizer
with learning rate `1e-4`, betas `(0.5, 0.999)`, and cosine annealing to
`1e-6`. These additions do not alter the generator reconstruction recipe or
generator optimizer.

Restoration uses the generator alone. The discriminator affects training cost
but is not required for generator inference.

## Metric implementation

The LPIPS training object uses the standard learned perceptual implementation.
Validation uses the local IQA-PyTorch snapshot and separately stored metric
weights. The AutoDL snapshot reported version `0.0.0`, but its exact upstream
commit was not preserved; this is a dependency-provenance limitation rather
than a substituted version claim.

The historical metric wrapper can retain zero defaults after metric exceptions.
The completed Stage III CSVs are non-zero, but per-image exception counts were
not logged.

## Reproducibility boundary

The comparison is single-run. Dataset, split, loader behavior, initialization
and training length are matched, but the historical run did not record a seed
or consumed-sample manifest. The sampler also updates its paths after the
dataset captures its initial lists. Therefore batch-for-batch identity is not
claimed.

The ten full-validation records, endpoint values, source hashes and checkpoint
hashes are published under `results/stage3/` so that every reported repository
claim can be checked without downloading the large training artifacts.
