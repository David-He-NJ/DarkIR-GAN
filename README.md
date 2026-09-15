# DarkIR-GAN

Official training code for **DarkIR-GAN**, a low-light image restoration model built on the DarkIR architecture with adversarial training.

> This repository currently contains training code and configuration files only. Datasets, pretrained models, checkpoints, and experiment outputs are not included.

## Repository structure

```text
DarkIR-GAN/
├── training/
│   ├── baseline/             # Initial DarkIR baseline training and evaluation
│   ├── baseline_continued/   # Continued baseline training from a checkpoint
│   └── gan/                  # DarkIR-GAN adversarial training
├── .gitignore
└── requirements.txt
```

Each training directory is self-contained and includes its own model, data, loss, and configuration modules. This preserves the exact code used for each experimental stage.

## Installation

Python 3.10 or later is recommended.

```bash
git clone git@github.com:David-He-NJ/DarkIR-GAN.git
cd DarkIR-GAN
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation

Download the LOL-Blur dataset and organize it as follows:

```text
datasets/
└── LOL-Blur/
    ├── train/
    └── val/
```

The default paths are defined in each `options/config.yaml`. Update them if your dataset is stored elsewhere.

## Training

### 1. Baseline training

```bash
cd training/baseline
python train_baseline.py
```

Alternative baseline entry points are retained for reproducibility:

- `train_initial.py`: initial baseline training routine.
- `train_from_pretrained.py`: training initialized from pretrained DarkIR weights.
- `train_with_periodic_evaluation.py`: training with periodic metric evaluation.
- `train_baseline_variant.py`: the additional baseline training variant used during development.
- `evaluate.py`: evaluation on the validation set.

### 2. Continued baseline training

Set `resume_path` in `training/baseline_continued/options/config.yaml`, then run:

```bash
cd training/baseline_continued
python train_continued.py
```

### 3. DarkIR-GAN training

GAN training initializes the generator from the baseline checkpoint at:

```text
training/baseline/experiments/checkpoints/baseline_best_lpips.pt
```

Then run:

```bash
cd training/gan
python train_gan.py
```

See `training/gan/MODEL_ARCHITECTURE.md` for additional architecture notes.

## Pretrained IQA weights

The training scripts expect optional IQA model files under:

```text
weights/iqa/
├── lpips.pth
├── musiq.pth
├── niqe.mat
└── nrqm.mat
```

These files are intentionally excluded from version control.

## Outputs

Checkpoints, TensorBoard logs, CSV logs, and validation visualizations are written to the local `experiments/` directory inside each training stage. These generated artifacts are excluded from Git.

## Citation

Citation information will be added after the paper is accepted or published.
