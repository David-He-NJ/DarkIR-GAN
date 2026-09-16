# Implementation and manuscript consistency

This audit describes the research snapshot. No model training, inference or metric recomputation was performed. Syntax validation is not numerical validation.

## Effective settings

Loss classes apply YAML weights and training loops apply additional multipliers.

| Setting | Baseline continued | GAN |
|---|---|---|
| Pixel coefficient | 1 | 1 |
| LPIPS coefficient | 0.1 × 0.1 = 0.01 | 1 |
| Laplacian coefficient | 5 × 5 = 25 | 0.1 × 0.1 = 0.01 |
| Enhancement | 0.25 × E(output,target) | C(input,target): zero generator gradient |
| Adversarial coefficient | 0 | 0.01 |
| Generator optimizer | AdamW, betas (0.9,0.999) | Adam, betas (0.5,0.999) |
| Generator learning rate | 1e-4 | 1e-5 |
| Training-output clamp | [0,1] before loss | No matching clamp |

Baseline E combines image L1 and 0.01 times VGG19 feature L1. GAN C is mean absolute input–target error: a logged offset with no generator-parameter dependence. Original CSV loss values must not be silently rewritten.

The GAN function argument, configuration and printouts do not all match the hardcoded 0.01 coefficient actually used. Some script constants also override YAML optimizer fields.

## LPIPS

The standard [LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py) maps net='vgg' to VGG16 and includes feature normalization and learned calibration. It defaults to normalize=False; normalize=True converts [0,1] inputs to the expected range.

The historical loss call passes tensors directly. Validation uses a separate pyiqa call and local weights, so a training-loss issue alone does not establish that validation LPIPS is wrong. Package versions, weight provenance and evaluation behavior still need checking. A normalization fix would define a different training recipe and cannot be described as producing the old results.

## Existing evidence

- The three manuscript CSVs match training-directory copies byte-for-byte: ten records each, epochs 3–30.
- Stage-II best_psnr and best_lpips archives contain identical serialized metadata and tensor-storage payloads after archive-root normalization. The GAN initialization copy also matches. Outer archive hashes differ.
- This verifies available files, not exact historical process arguments.
- Endpoint tables use epoch 30, with differences computed before rounding. GAN NRQM rounds to 5.821; its baseline difference to +0.621. Stage-II final NIQE rounds to 5.561.
- Stage-I numerical results and cross-method SOTA numbers lack matching evaluation records in the supplied bundle and are not presented as reproduced results.

The current evidence supports a comparison between two recipes. Matching available initialization does not remove differences in losses, clipping, optimizers, learning rates or sampling. A causal attribution to adversarial supervision requires a matched ablation.

Restoration uses the generator alone. Discriminator training cost does not establish additional generator inference cost.

## Controlled GAN setup

`training/gan_controlled/` defines the missing matched ablation. It imports the actual `baseline_continued` generator, dataloader, loss construction, generator optimizer/scheduler, and validation functions. Its startup guard requires the generator, train, dataset, and initialization-path configuration to remain aligned with the continued baseline.

The controlled generator keeps the continued baseline's runtime overrides and effective reconstruction objective, including output clamping and its existing `E(output,target)` enhancement term. This is not a correction of the historical GAN's `C(input,target)` term; the zero-generator-gradient historical expression is absent. PatchGAN/LSGAN with coefficient `0.01` is the only added generator-objective term.

The controlled run restores generator weights, optimizer state, and scheduler state from `baseline_best_psnr.pt`, fails closed if that checkpoint is unavailable, and records its SHA-256 plus the restored optimizer learning rates and scheduler epoch in a run manifest. Because optimizer state restoration occurs after construction, the first-step generator learning rate is whatever the checkpoint stores; `1e-4` is the matching construction-time override, not a guarantee about the restored state. The setup has not been trained in this repository revision.

Exact batch-for-batch equivalence to the historical continued-baseline run still cannot be guaranteed. The existing sampler resamples internal paths after the dataset has captured its initial path lists, and the historical run provides no seed or consumed-sample manifest. This behavior is intentionally preserved instead of silently introducing a second experimental change.
