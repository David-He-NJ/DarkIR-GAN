# Controlled GAN experiment

This directory defines the matched adversarial ablation requested after the implementation audit. It uses `training/baseline_continued/` as the authoritative generator recipe and imports its generator, dataloader, reconstruction losses, optimizer factory, scheduler, validation, and metric logging directly.

## Controlled question

The intended comparison is:

```text
baseline_continued generator objective
vs.
baseline_continued generator objective + 0.01 * PatchGAN LSGAN loss
```

The generator-side settings are guarded at startup against the current continued-baseline YAML. A mismatch aborts before any training. The continued-baseline runtime construction is also preserved: 30 epochs, generator AdamW constructed with learning rate `1e-4`, betas `(0.9, 0.999)`, weight decay `1e-5`, cosine annealing to `1e-6`, and no gradient clipping. As in `baseline_continued`, the shared checkpoint then restores the generator optimizer and scheduler states; therefore the actual first-step learning rate is the value stored in that checkpoint, not necessarily `1e-4`.

The reconstruction objective intentionally keeps the continued baseline's current output clamp and effective coefficients:

| Component | Effective coefficient / behavior |
|---|---|
| Pixel L1 | `1` |
| LPIPS object | YAML `0.1` times outer `0.1` = `0.01` |
| Laplacian edge object | YAML `5` times outer `5` = `25` |
| Baseline enhancement object | YAML `0.5` times outer `0.5` = `0.25`, evaluated on `(output, target)` |
| Adversarial | LSGAN MSE, outer weight `0.01` |

The enhancement line is not a correction of the historical GAN's `enhance_loss(inputs, targets)` expression. That zero-generator-gradient expression is absent. The `(output, target)` call exists here only because it is already part of the actual `baseline_continued` objective; removing or replacing it would create another variable.

## Added adversarial components

- Five-layer PatchGAN discriminator, matching the topology of the historical GAN directory.
- LSGAN/MSE discriminator and generator objectives.
- Real-label smoothing to `0.9`, fake label `0.0`.
- Discriminator-only Adam optimizer: learning rate `1e-4`, betas `(0.5, 0.999)`, cosine annealing to `1e-6`.

These discriminator settings have no baseline counterpart. They are part of the adversarial treatment and do not change the generator reconstruction recipe or generator optimizer.

## Initialization and run records

The run requires the same default initialization path as the continued baseline:

```text
training/baseline/experiments/checkpoints/baseline_best_psnr.pt
```

Unlike the historical scripts, a missing or incomplete checkpoint is a hard error; the controlled run never silently falls back to training from scratch. Generator weights, generator optimizer state, and generator scheduler state are all restored exactly as `baseline_continued` does. The script verifies `intro.weight`, computes the checkpoint SHA-256, and writes it with config/source hashes, restored learning rates, and scheduler epoch to `experiments/run_manifest.json`.

Run only after preparing the documented CUDA environment, dataset, IQA weights, and initialization checkpoint:

```bash
(cd training/gan_controlled && python train_controlled_gan.py)
```

No training is started by repository setup or verification. `verify_setup.py` is a standard-library-only static guard:

```bash
python training/gan_controlled/verify_setup.py
python -m compileall -q training/gan_controlled
```

## Remaining fairness limitation

The existing scene sampler chooses an initial subset before dataset construction. Later `set_epoch()` calls update sampler paths, but the dataset retains the initially captured path lists. The controlled code preserves this behavior to avoid changing the recipe, but a new run cannot prove it consumed the same initial random subset/order as the historical continued-baseline run because that run did not record a seed or manifest. Dataset/split and loader behavior are matched; batch-for-batch identity is not guaranteed.

Evaluation should be rerun for both final checkpoints with one strict evaluator. The legacy metric helper catches exceptions and records zeros, so its aggregate values alone are not suitable evidence of a clean matched comparison.
