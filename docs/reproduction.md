# Reproduction requirements

## Verified scope

The remote upload, Python syntax and YAML parsing were checked. Documentation was compared with the manuscript, source and aggregate CSVs. No clean-environment GPU execution or metric recomputation was performed.

## Historical stages

1. Stage I is described as one full-data epoch. train_from_pretrained.py and train_with_periodic_evaluation.py request full training images but use configuration-driven epoch counts. Neither currently has a dedicated one-epoch configuration. The original entry point and Stage-I log remain to be confirmed.
2. Stage II uses a baseline initialization and the baseline training routines. An additional variant is retained; selecting an arbitrary variant does not prove reproduction of the CSV.
3. Stage III has separate baseline_continued and gan branches, with different effective settings.

## Before future execution

- Recover original Python/CUDA/PyTorch/torchvision/LPIPS/pyiqa versions and metric-weight hashes. requirements.txt is not a version lock.
- Validate paired filenames, dimensions and full/quick-validation counts. The loader does not enforce the previously claimed 256×256 evaluation resolution.
- Ensure initialization files exist; missing paths can trigger from-scratch fallback.
- Freeze the chosen source revision and effective runtime settings, including script overrides.
- Check actual sampling: the sampler updates its own paths, but the dataset retains paths captured at construction. set_epoch alone does not establish new images were consumed.
- Check LPIPS conventions separately for training loss and the validation wrapper.
- Record metric failures. The historical wrapper catches exceptions and retains zero defaults; aggregate logs do not report failure counts.
- Check checkpoint formats. baseline/evaluate.py is not a general GAN evaluator.

## Remaining evidence

Matched ablations, repeated-seed variation, failure counts and clean-environment reproduction cannot be established by editing prose. Some evaluation questions may be answered from old outputs/checkpoints without retraining, but no such evaluation was run in this revision.

## Controlled adversarial ablation

The matched setup is under `training/gan_controlled/`. Before allocating GPU time:

```bash
python training/gan_controlled/verify_setup.py
python -m compileall -q training/gan_controlled
```

Then place the shared Stage-II initialization at `training/baseline/experiments/checkpoints/baseline_best_psnr.pt` and launch from the controlled directory. The run aborts rather than falling back to random initialization and records the initialization SHA-256 in `training/gan_controlled/experiments/run_manifest.json`.

After training, run baseline and controlled-GAN generator checkpoints through the same strict evaluator. Record metric failure counts; do not use silent zero substitution as evidence. The historical sampler does not provide proof of identical batch order, so describe the comparison as matched in dataset/split, loader behavior, training length, and generator recipe—not as batch-identical.

Historical code semantics and logs are preserved. Future fixes and measurements should use a separately identified experiment revision.
