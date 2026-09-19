"""Controlled GAN continuation of training/baseline_continued.

The generator, data loader, reconstruction losses, generator optimizer,
scheduler, validation, and initialization are imported from the continued
baseline. The only generator-objective addition is PatchGAN adversarial loss.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import pyiqa
import yaml


CONTROLLED_DIR = Path(__file__).resolve().parent
BASELINE_DIR = CONTROLLED_DIR.parent / "baseline_continued"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

# Import the actual baseline implementation rather than maintaining another
# copy of its generator/data/loss/evaluation modules.
import train_continued as baseline  # noqa: E402

from patchgan import PatchGANDiscriminator  # noqa: E402


TOTAL_EPOCHS = 30
GENERATOR_LR = 1e-4
ETA_MIN = 1e-6

# These are the outer multipliers hardcoded by baseline_continued. The loss
# objects also apply their YAML weights; preserving both layers is intentional.
PERCEPTUAL_OUTER_WEIGHT = 0.1
EDGE_OUTER_WEIGHT = 5.0
ENHANCE_OUTER_WEIGHT = 0.5


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_baseline_alignment(controlled_opt, baseline_opt):
    """Fail before training if a baseline-controlled field has drifted."""
    compared_sections = ("network", "train", "datasets")
    mismatches = [
        section
        for section in compared_sections
        if controlled_opt.get(section) != baseline_opt.get(section)
    ]
    if controlled_opt.get("path", {}).get("resume_path") != baseline_opt.get("path", {}).get("resume_path"):
        mismatches.append("path.resume_path")
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            "Controlled GAN configuration drifted from baseline_continued in: "
            f"{joined}. Only discriminator/adversarial settings may differ."
        )


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def reconstruction_loss(criterion, outputs, targets):
    """Reproduce baseline_continued's effective generator objective exactly."""
    loss_pixel = criterion["pixel_loss"](outputs, targets)

    loss_perceptual = 0.0
    if "perceptual_loss" in criterion:
        loss_perceptual = criterion["perceptual_loss"](outputs, targets)

    loss_edge = 0.0
    if "edge_loss" in criterion:
        loss_edge = criterion["edge_loss"](outputs, targets)

    # This is copied from baseline_continued: output-target, not the historical
    # GAN's zero-gradient input-target term and not a new corrected loss.
    loss_enhance = 0.0
    if "enhance_loss" in criterion:
        loss_enhance = criterion["enhance_loss"](outputs, targets)

    total = loss_pixel
    if not isinstance(loss_perceptual, float):
        total = total + PERCEPTUAL_OUTER_WEIGHT * loss_perceptual
    if not isinstance(loss_edge, float):
        total = total + EDGE_OUTER_WEIGHT * loss_edge
    if not isinstance(loss_enhance, float):
        total = total + ENHANCE_OUTER_WEIGHT * loss_enhance

    return total, loss_pixel, loss_perceptual, loss_edge, loss_enhance


def train_one_epoch(
    generator,
    discriminator,
    dataloader,
    criterion,
    optimizer_g,
    optimizer_d,
    device,
    adversarial_opt,
):
    generator.train()
    discriminator.train()
    adversarial_criterion = nn.MSELoss()

    totals = {
        "g": 0.0,
        "d": 0.0,
        "adv": 0.0,
        "pixel": 0.0,
        "perceptual": 0.0,
        "edge": 0.0,
        "enhance": 0.0,
    }

    for inputs, targets in baseline.tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = torch.clamp(targets.to(device), 0.0, 1.0)

        # Baseline clears G gradients before its forward pass and clamps the
        # generator output before every reconstruction loss.
        optimizer_g.zero_grad()
        fake_images = torch.clamp(generator(inputs), 0.0, 1.0)

        optimizer_d.zero_grad()
        real_logits = discriminator(targets)
        fake_logits = discriminator(fake_images.detach())
        real_labels = torch.full_like(real_logits, adversarial_opt["real_label"])
        fake_labels = torch.full_like(fake_logits, adversarial_opt["fake_label"])
        loss_d_real = adversarial_criterion(real_logits, real_labels)
        loss_d_fake = adversarial_criterion(fake_logits, fake_labels)
        loss_d = 0.5 * (loss_d_real + loss_d_fake)
        loss_d.backward()
        optimizer_d.step()

        set_requires_grad(discriminator, False)
        reconstruction, loss_pixel, loss_perceptual, loss_edge, loss_enhance = reconstruction_loss(
            criterion, fake_images, targets
        )
        generator_logits = discriminator(fake_images)
        generator_labels = torch.ones_like(generator_logits)
        loss_adv = adversarial_criterion(generator_logits, generator_labels)
        loss_g = reconstruction + adversarial_opt["weight"] * loss_adv
        loss_g.backward()
        optimizer_g.step()
        set_requires_grad(discriminator, True)

        totals["g"] += loss_g.item()
        totals["d"] += loss_d.item()
        totals["adv"] += loss_adv.item()
        totals["pixel"] += loss_pixel.item()
        totals["perceptual"] += 0.0 if isinstance(loss_perceptual, float) else loss_perceptual.item()
        totals["edge"] += 0.0 if isinstance(loss_edge, float) else loss_edge.item()
        totals["enhance"] += 0.0 if isinstance(loss_enhance, float) else loss_enhance.item()

    count = len(dataloader)
    if count == 0:
        raise RuntimeError("Training dataloader is empty.")
    averages = {name: value / count for name, value in totals.items()}
    print(
        "Epoch Training Loss: "
        f"G={averages['g']:.4f}, D={averages['d']:.4f}, "
        f"Pixel={averages['pixel']:.4f}, Perc={averages['perceptual']:.4f}, "
        f"Edge={averages['edge']:.4f}, Enhance={averages['enhance']:.4f}, "
        f"Adv={averages['adv']:.4f}"
    )
    return averages


def load_initialization(generator, optimizer_g, scheduler_g, checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Required shared baseline initialization is missing: {checkpoint_path}. "
            "Controlled training will not fall back to training from scratch."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"Initialization checkpoint is missing required keys: {missing}")

    generator.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer_g.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_g.load_state_dict(checkpoint["scheduler_state_dict"])

    probe = generator.state_dict()["intro.weight"]
    expected = checkpoint["model_state_dict"]["intro.weight"]
    if not torch.equal(probe.detach().cpu(), expected.detach().cpu()):
        raise RuntimeError("Generator initialization verification failed for intro.weight.")

    return checkpoint, sha256_file(checkpoint_path)


def build_discriminator_optimizer(discriminator, config):
    if config["optimizer"].lower() != "adam":
        raise ValueError("Only Adam is implemented for the controlled PatchGAN discriminator.")
    optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config["lr_initial"],
        betas=tuple(config["betas"]),
        weight_decay=config["weight_decay"],
    )
    if config["lr_scheme"] != "CosineAnnealing":
        raise ValueError("Only CosineAnnealing is implemented for the discriminator.")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TOTAL_EPOCHS,
        eta_min=config["eta_min"],
    )
    return optimizer, scheduler


def write_run_manifest(
    experiment_dir,
    checkpoint_path,
    checkpoint_sha256,
    config_path,
    optimizer_g,
    scheduler_g,
):
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "controlled_gan",
        "primary_variable": "PatchGAN adversarial training",
        "initialization_path": str(checkpoint_path),
        "initialization_sha256": checkpoint_sha256,
        "controlled_config_sha256": sha256_file(config_path),
        "baseline_config_sha256": sha256_file(BASELINE_DIR / "options" / "config.yaml"),
        "baseline_train_script_sha256": sha256_file(BASELINE_DIR / "train_continued.py"),
        "generator_runtime_overrides": {
            "epochs": TOTAL_EPOCHS,
            "lr_initial": GENERATOR_LR,
            "eta_min": ETA_MIN,
        },
        "generator_optimizer_lrs_after_restore": [
            group["lr"] for group in optimizer_g.param_groups
        ],
        "generator_scheduler_last_epoch_after_restore": scheduler_g.last_epoch,
    }
    output_path = experiment_dir / "run_manifest.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def save_checkpoint(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
    epoch,
    val_psnr,
    val_lpips,
    save_dir,
    best_psnr,
    best_lpips,
    initialization_sha256,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        # Keep the baseline key so the same generator evaluator can load it.
        "model_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_state_dict": optimizer_g.state_dict(),
        "optimizer_D_state_dict": optimizer_d.state_dict(),
        "scheduler_state_dict": scheduler_g.state_dict(),
        "scheduler_D_state_dict": scheduler_d.state_dict(),
        "val_psnr": val_psnr,
        "val_lpips": val_lpips,
        "initialization_sha256": initialization_sha256,
        "experiment": "controlled_gan",
    }

    periodic_path = save_dir / f"controlled_gan_epoch_{epoch}.pt"
    torch.save(checkpoint, periodic_path)
    print(f"Periodic checkpoint saved: {periodic_path}")

    if val_psnr > best_psnr:
        path = save_dir / "controlled_gan_best_psnr.pt"
        torch.save(checkpoint, path)
        best_psnr = val_psnr
        print(f"New best PSNR {val_psnr:.2f}: {path}")
    if val_lpips < best_lpips:
        path = save_dir / "controlled_gan_best_lpips.pt"
        torch.save(checkpoint, path)
        best_lpips = val_lpips
        print(f"New best LPIPS {val_lpips:.4f}: {path}")
    return best_psnr, best_lpips


def main():
    os.chdir(CONTROLLED_DIR)
    config_path = CONTROLLED_DIR / "options" / "config.yaml"
    controlled_opt = load_yaml(config_path)
    baseline_opt = load_yaml(BASELINE_DIR / "options" / "config.yaml")
    assert_baseline_alignment(controlled_opt, baseline_opt)

    if controlled_opt["adversarial"]["criterion"].lower() != "mse":
        raise ValueError("Only the configured LSGAN/MSE objective is implemented.")
    if not torch.cuda.is_available():
        raise RuntimeError("The inherited baseline model factory requires CUDA.")

    device = torch.device("cuda")
    rank = 0

    # Match the effective overrides in baseline_continued/train_continued.py.
    controlled_opt["train"]["epochs"] = TOTAL_EPOCHS
    controlled_opt["train"]["lr_initial"] = GENERATOR_LR
    controlled_opt["train"]["eta_min"] = ETA_MIN

    print("Controlled GAN configuration")
    print(f"  epochs: {TOTAL_EPOCHS}")
    print(f"  generator: AdamW lr={GENERATOR_LR}, betas={tuple(controlled_opt['train']['betas'])}")
    print(f"  adversarial weight: {controlled_opt['adversarial']['weight']}")
    print("  generator recipe: baseline_continued + adversarial term only")

    generator, _, _ = baseline.create_model(controlled_opt["network"], rank)
    discriminator = PatchGANDiscriminator(
        input_channels=controlled_opt["discriminator"]["input_channels"],
        base_channels=controlled_opt["discriminator"]["base_channels"],
    ).to(device)

    criterion = baseline.create_loss(controlled_opt["train"]["loss"], rank=rank)
    optimizer_g, scheduler_g = baseline.create_optim_scheduler(controlled_opt["train"], generator)
    optimizer_d, scheduler_d = build_discriminator_optimizer(
        discriminator, controlled_opt["discriminator"]
    )

    checkpoint_path = (CONTROLLED_DIR / controlled_opt["path"]["resume_path"]).resolve()
    initialization, initialization_sha256 = load_initialization(
        generator, optimizer_g, scheduler_g, checkpoint_path, device
    )
    print(f"Loaded shared initialization: {checkpoint_path}")
    print(f"Initialization SHA-256: {initialization_sha256}")
    print(
        "Restored generator learning rate(s): "
        f"{[group['lr'] for group in optimizer_g.param_groups]}"
    )
    print(f"Restored generator scheduler last_epoch: {scheduler_g.last_epoch}")

    print("Loading IQA metrics...")
    lpips_model = pyiqa.create_metric(
        "lpips", device=device, pretrained_model_path="../../weights/iqa/lpips.pth"
    )
    musiq_model = pyiqa.create_metric(
        "musiq", device=device, pretrained_model_path="../../weights/iqa/musiq.pth"
    )
    niqe_model = pyiqa.create_metric(
        "niqe", device=device, pretrained_model_path="../../weights/iqa/niqe.mat"
    )
    nrqm_model = pyiqa.create_metric(
        "nrqm", device=device, pretrained_model_path="../../weights/iqa/nrqm.mat"
    )

    train_loader, train_sampler = baseline.create_dataloaders(
        rank, 1, controlled_opt["datasets"], mode="train", return_path=False, samples_per_scene=15
    )
    val_loader_quick, _ = baseline.create_dataloaders(
        rank, 1, controlled_opt["datasets"], mode="val", return_path=False, samples_per_scene=15
    )
    val_loader_full, _ = baseline.create_dataloaders(
        rank, 1, controlled_opt["datasets"], mode="val", return_path=True, samples_per_scene=-1
    )

    experiment_dir = CONTROLLED_DIR / "experiments"
    vis_save_dir = experiment_dir / "ControlledGAN_vis"
    save_dir = experiment_dir / "checkpoints"
    log_path = experiment_dir / "controlled_gan_training_log.csv"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    vis_save_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    baseline.init_csv_logger(str(log_path))
    writer = SummaryWriter(log_dir=str(experiment_dir / "tensorboard_controlled_gan"))
    manifest_path = write_run_manifest(
        experiment_dir,
        checkpoint_path,
        initialization_sha256,
        config_path,
        optimizer_g,
        scheduler_g,
    )
    print(f"Run manifest: {manifest_path}")

    best_psnr = initialization.get("val_psnr", 0.0)
    best_lpips = initialization.get("val_lpips", float("inf"))
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    baseline.validate_quick(
        generator,
        val_loader_quick,
        criterion,
        device,
        psnr_metric,
        ssim_metric,
        lpips_model,
        musiq_model,
        nrqm_model,
        niqe_model,
        0,
        desc="Init Val (Loaded)",
    )

    try:
        for epoch in range(TOTAL_EPOCHS):
            print(f"\nEpoch [{epoch + 1}/{TOTAL_EPOCHS}]")
            if hasattr(train_sampler, "set_epoch"):
                # Preserve baseline behavior, including its known sampler/path limitation.
                train_sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                generator,
                discriminator,
                train_loader,
                criterion,
                optimizer_g,
                optimizer_d,
                device,
                controlled_opt["adversarial"],
            )
            scheduler_g.step()
            scheduler_d.step()

            quick = baseline.validate_quick(
                generator,
                val_loader_quick,
                criterion,
                device,
                psnr_metric,
                ssim_metric,
                lpips_model,
                musiq_model,
                nrqm_model,
                niqe_model,
                epoch,
                desc=f"Epoch {epoch + 1} Quick Val",
            )

            for key, value in train_metrics.items():
                writer.add_scalar(f"Loss/train_{key}", value, epoch + 1)
            writer.add_scalar("Loss/val_quick_total", quick[0], epoch + 1)
            writer.add_scalar("Metrics/PSNR_quick", quick[5], epoch + 1)
            writer.add_scalar("Metrics/SSIM_quick", quick[6], epoch + 1)
            writer.add_scalar("Metrics/LPIPS_quick", quick[7], epoch + 1)

            if (epoch + 1) % 3 == 0 or epoch == TOTAL_EPOCHS - 1:
                full = baseline.validate(
                    generator,
                    val_loader_full,
                    criterion,
                    device,
                    psnr_metric,
                    ssim_metric,
                    lpips_model,
                    musiq_model,
                    nrqm_model,
                    niqe_model,
                    epoch,
                    str(vis_save_dir),
                )
                baseline.append_csv_log(
                    str(log_path), epoch + 1, *full
                )
                writer.add_scalar("Loss/val_full_total", full[0], epoch + 1)
                writer.add_scalar("Metrics/PSNR_full", full[5], epoch + 1)
                writer.add_scalar("Metrics/SSIM_full", full[6], epoch + 1)
                writer.add_scalar("Metrics/LPIPS_full", full[7], epoch + 1)
                best_psnr, best_lpips = save_checkpoint(
                    generator,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    scheduler_g,
                    scheduler_d,
                    epoch + 1,
                    full[5],
                    full[7],
                    save_dir,
                    best_psnr,
                    best_lpips,
                    initialization_sha256,
                )

            torch.cuda.empty_cache()
    finally:
        writer.close()

    print("Controlled GAN training completed.")
    print(f"Best PSNR: {best_psnr:.2f}")
    print(f"Best LPIPS: {best_lpips:.4f}")


if __name__ == "__main__":
    main()
