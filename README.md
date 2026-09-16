# DarkIR-GAN

**DarkIR-GAN: GAN-Based Low-Light Blurred Image Restoration**

Xinyu He and Shi Yan — School of Information Science and Engineering, Lanzhou University.

Research code for comparing continued DarkIR training with a perception-oriented GAN fine-tuning recipe. The model combines a DarkIR generator with a five-layer PatchGAN discriminator. Restoration inference uses only the generator.

## Status

This research snapshot has not undergone a clean-environment GPU reproduction test. The historical comparison changes reconstruction weights and optimization settings as well as adversarial supervision, so it does not isolate the causal effect of the adversarial term. `training/gan_controlled/` adds a matched experiment setup; no results are claimed until that setup is trained and evaluated.

See [implementation notes](docs/implementation_notes.md) and [reproduction requirements](docs/reproduction.md).

## Workflow and structure

```text
Stage I: full-data adaptation (historical entry points in baseline/)
    |
Stage II: baseline training (baseline/)
    |
    +-- Stage III-A: continued baseline (baseline_continued/)
    +-- Stage III-B: historical GAN fine-tuning (gan/)
    +-- Controlled GAN: Stage III-A recipe + adversarial training only (gan_controlled/)
```

These experiment directories are not four consecutive stages. Stage I has no dedicated one-epoch configuration in the uploaded snapshot.

```text
training/
├── baseline/
│   ├── archs/, data/, losses/, options/
│   ├── train_with_periodic_evaluation.py
│   ├── train_from_pretrained.py
│   ├── train_initial.py
│   ├── train_baseline.py
│   ├── train_baseline_variant.py
│   └── evaluate.py
├── baseline_continued/
│   └── train_continued.py
├── gan/
│   ├── train_gan.py
│   ├── train_baseline.py
│   └── MODEL_ARCHITECTURE.md
└── gan_controlled/
    ├── train_controlled_gan.py
    ├── patchgan.py
    ├── verify_setup.py
    └── README.md
```

Stage-specific module copies are retained because implementations differ. The historical `gan/train_baseline.py` has a model-factory call incompatible with the GAN package; it is not the baseline launch command.

## Environment

Install a CUDA-compatible PyTorch/torchvision pair and the dependencies in an isolated environment. `requirements.txt` was extracted from imports; it is not the original version lock, and no fresh installation has validated a version combination. Some code paths explicitly require CUDA.

```bash
git clone git@github.com:David-He-NJ/DarkIR-GAN.git
cd DarkIR-GAN
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data and weights

The configured layout, relative to the repository root, is:

```text
datasets/LOL-Blur/
├── train/
│   ├── low_blur/<scene_id>/<image>.png
│   └── high_sharp_original/<scene_id>/<image>.png
└── val/
    ├── low_blur/<scene_id>/<image>.png
    └── high_sharp_original/<scene_id>/<image>.png
```

Validate pairing and dimensions before execution. The loader does not force 256×256 resizing. `weights/iqa/` must contain compatible `lpips.pth`, `musiq.pth`, `niqe.mat` and `nrqm.mat` files: they are required by the metric calls, not optional assets. Loss libraries may additionally load pretrained VGG/LPIPS weights.

Datasets, weights and experiment outputs are excluded from Git.

## Entry points

Each command starts from the **repository root**, after preparing data, environment, metric weights and initialization. These commands identify entry points; they are not a guarantee of unverified end-to-end execution.

```bash
# Stage II: inspect resume_path and script overrides first.
(cd training/baseline && python train_baseline.py)

# Stage III-A.
(cd training/baseline_continued && python train_continued.py)

# Stage III-B.
(cd training/gan && python train_gan.py)

# Matched controlled GAN (requires the shared baseline checkpoint).
(cd training/gan_controlled && python train_controlled_gan.py)
```

Stage III-A defaults to `training/baseline/experiments/checkpoints/baseline_best_psnr.pt`; Stage III-B defaults to the sibling `baseline_best_lpips.pt`. Missing checkpoints can trigger from-scratch training, so check them explicitly.

The two supplied Stage-II archives have identical serialized metadata and tensor-storage contents despite different file hashes. This supports a shared available initialization but does not prove what a past process loaded.

The controlled entry point refuses to train if its baseline-controlled YAML sections drift or if the shared initialization is missing. It records the initialization SHA-256 and source/config hashes in a run manifest. See [the controlled experiment notes](training/gan_controlled/README.md).

`baseline/evaluate.py` is a baseline evaluator, not a unified GAN evaluator. A standalone inference interface is not included. Outputs are saved under each stage's `experiments/` directory.

## Implemented objective

Effective GAN coefficients, after including both inner and outer multipliers:

- Pixel L1: 1.
- LPIPS: 1.
- Laplacian residual MSE: 0.01.
- LSGAN generator loss: 0.01.

A historical input–target L1 term contributes to the logged loss but has zero generator gradient. The baseline uses a different reconstruction recipe. See the [full accounting](docs/implementation_notes.md).

The controlled setup does not reuse that zero-gradient term. It retains the actual continued-baseline reconstruction objective and adds only a `0.01` LSGAN generator term. This is an experiment definition, not a result.

## Acknowledgements and citation

The generator follows DarkIR; the framework uses PatchGAN/LSGAN concepts and LPIPS-based losses. The manuscript bibliography identifies the underlying work. No acceptance status or venue is asserted. A project license has not been selected; this snapshot does not grant a new blanket license to upstream components.
