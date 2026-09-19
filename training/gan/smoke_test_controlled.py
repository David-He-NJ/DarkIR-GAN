"""Bounded runtime smoke test for the controlled GAN experiment.

This harness imports the production implementation from the pinned GitHub
snapshot. It never enters the 30-epoch production loop. Its only outputs live
under ``experiments/smoke`` and are safe to remove after inspection.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import pyiqa
import torch
import torchvision
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINE_DIR = HERE.parent / "baseline_continued"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import train_controlled_gan as controlled  # noqa: E402


class LimitedLoader:
    def __init__(self, loader, limit):
        self.loader = loader
        self.limit = min(limit, len(loader))
        self.memory_samples = []

    def __iter__(self):
        for index, batch in enumerate(itertools.islice(iter(self.loader), self.limit)):
            yield batch
            torch.cuda.synchronize()
            self.memory_samples.append(
                {
                    "iteration": index + 1,
                    "allocated_gib": torch.cuda.memory_allocated() / (1024 ** 3),
                    "reserved_gib": torch.cuda.memory_reserved() / (1024 ** 3),
                }
            )

    def __len__(self):
        return self.limit


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} is not finite: {number}")
    return number


def tensor_equal(left, right):
    return torch.equal(left.detach().cpu(), right.detach().cpu())


def gradient_summary(module, name):
    gradients = [
        parameter.grad.detach()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError(f"{name} has no gradients")
    finite = all(torch.isfinite(gradient).all().item() for gradient in gradients)
    squared_norm = sum(
        float(torch.sum(gradient.float() * gradient.float()).item())
        for gradient in gradients
    )
    norm = math.sqrt(squared_norm)
    nonzero_tensors = sum(int(torch.count_nonzero(gradient).item() > 0) for gradient in gradients)
    if not finite or not math.isfinite(norm) or norm <= 0 or nonzero_tensors == 0:
        raise RuntimeError(
            f"{name} gradients invalid: finite={finite}, norm={norm}, "
            f"nonzero_tensors={nonzero_tensors}"
        )
    return {
        "finite": finite,
        "global_l2_norm": norm,
        "tensors_present": len(gradients),
        "nonzero_tensors": nonzero_tensors,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--full-train-batches", type=int, default=2550)
    parser.add_argument("--full-quick-validation-batches", type=int, default=450)
    parser.add_argument("--full-validation-batches", type=int, default=1800)
    parser.add_argument("--run-name", default="smoke")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 100 <= args.iterations <= 500:
        raise ValueError("--iterations must be between 100 and 500")
    if args.validation_batches < 1:
        raise ValueError("--validation-batches must be positive")
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--run-name may contain only letters, numbers, hyphens, and underscores")

    os.chdir(HERE)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")

    controlled_opt = controlled.load_yaml(HERE / "options" / "config.yaml")
    baseline_opt = controlled.load_yaml(BASELINE_DIR / "options" / "config.yaml")
    controlled.assert_baseline_alignment(controlled_opt, baseline_opt)
    if args.dataset_root is not None:
        dataset_root = args.dataset_root.resolve()
        controlled_opt["datasets"]["train"]["dataroot"] = str(dataset_root / "train")
        controlled_opt["datasets"]["val"]["dataroot"] = str(dataset_root / "val")
    controlled_opt["train"]["epochs"] = controlled.TOTAL_EPOCHS
    controlled_opt["train"]["lr_initial"] = controlled.GENERATOR_LR
    controlled_opt["train"]["eta_min"] = controlled.ETA_MIN

    generator, _, _ = controlled.baseline.create_model(controlled_opt["network"], 0)
    discriminator = controlled.PatchGANDiscriminator(
        input_channels=controlled_opt["discriminator"]["input_channels"],
        base_channels=controlled_opt["discriminator"]["base_channels"],
    ).to(device)
    criterion = controlled.baseline.create_loss(controlled_opt["train"]["loss"], rank=0)
    optimizer_g, scheduler_g = controlled.baseline.create_optim_scheduler(
        controlled_opt["train"], generator
    )
    optimizer_d, scheduler_d = controlled.build_discriminator_optimizer(
        discriminator, controlled_opt["discriminator"]
    )

    checkpoint_path = (HERE / controlled_opt["path"]["resume_path"]).resolve()
    initialization, checkpoint_sha256 = controlled.load_initialization(
        generator, optimizer_g, scheduler_g, checkpoint_path, device
    )
    initial_generator_lrs = [group["lr"] for group in optimizer_g.param_groups]
    initial_generator_scheduler_last_epoch = scheduler_g.last_epoch

    train_loader, train_sampler = controlled.baseline.create_dataloaders(
        0,
        1,
        controlled_opt["datasets"],
        mode="train",
        return_path=False,
        samples_per_scene=15,
    )
    val_loader, _ = controlled.baseline.create_dataloaders(
        0,
        1,
        controlled_opt["datasets"],
        mode="val",
        return_path=False,
        samples_per_scene=15,
    )
    full_val_loader, _ = controlled.baseline.create_dataloaders(
        0,
        1,
        controlled_opt["datasets"],
        mode="val",
        return_path=False,
        samples_per_scene=-1,
    )
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(0)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    monitored_train_loader = LimitedLoader(train_loader, args.iterations)
    train_metrics = controlled.train_one_epoch(
        generator,
        discriminator,
        monitored_train_loader,
        criterion,
        optimizer_g,
        optimizer_d,
        device,
        controlled_opt["adversarial"],
    )
    torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - train_start
    scheduler_g.step()
    scheduler_d.step()
    generator_gradients = gradient_summary(generator, "generator")
    discriminator_gradients = gradient_summary(discriminator, "discriminator")

    memory_samples = monitored_train_loader.memory_samples
    if len(memory_samples) != args.iterations:
        raise RuntimeError("Per-iteration GPU memory sampling was incomplete")
    warmup_start = min(9, len(memory_samples) - 1)
    post_warmup_memory = memory_samples[warmup_start:]
    allocated_values = [sample["allocated_gib"] for sample in post_warmup_memory]
    reserved_values = [sample["reserved_gib"] for sample in post_warmup_memory]
    memory_stability = {
        "warmup_ignored_iterations": warmup_start,
        "post_warmup_allocated_span_gib": max(allocated_values) - min(allocated_values),
        "post_warmup_reserved_span_gib": max(reserved_values) - min(reserved_values),
        "post_warmup_allocated_end_minus_start_gib": (
            allocated_values[-1] - allocated_values[0]
        ),
        "post_warmup_reserved_end_minus_start_gib": (
            reserved_values[-1] - reserved_values[0]
        ),
    }
    memory_stable = (
        abs(memory_stability["post_warmup_allocated_end_minus_start_gib"]) < 0.5
        and abs(memory_stability["post_warmup_reserved_end_minus_start_gib"]) < 0.5
        and memory_stability["post_warmup_allocated_span_gib"] < 1.0
        and memory_stability["post_warmup_reserved_span_gib"] < 1.0
    )
    memory_stability["stable"] = memory_stable
    if not memory_stable:
        raise RuntimeError(f"GPU memory did not stabilize: {memory_stability}")

    for name in ("g", "d", "adv", "pixel", "perceptual", "edge", "enhance"):
        finite_number(train_metrics[name], f"train_{name}")
    if train_metrics["adv"] <= 0:
        raise RuntimeError("Adversarial loss did not execute")
    if train_metrics["perceptual"] <= 0:
        raise RuntimeError("LPIPS/perceptual loss did not execute")

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
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    limited_val = LimitedLoader(val_loader, args.validation_batches)
    torch.cuda.synchronize()
    validation_start = time.perf_counter()
    quick = controlled.baseline.validate_quick(
        generator,
        limited_val,
        criterion,
        device,
        psnr_metric,
        ssim_metric,
        lpips_model,
        musiq_model,
        nrqm_model,
        niqe_model,
        0,
        desc="Smoke validation",
    )
    torch.cuda.synchronize()
    validation_elapsed = time.perf_counter() - validation_start

    # The legacy helper suppresses metric exceptions, so run every metric
    # explicitly and require a finite result from each call.
    sample_inputs, sample_targets = next(iter(val_loader))
    with torch.no_grad():
        sample_outputs = torch.clamp(generator(sample_inputs.to(device)), 0.0, 1.0)
        sample_targets = sample_targets.to(device)
        explicit_metrics = {
            "psnr": psnr_metric(sample_outputs, sample_targets),
            "ssim": ssim_metric(sample_outputs, sample_targets),
            "lpips": lpips_model(sample_outputs, sample_targets).mean(),
            "musiq": musiq_model(sample_outputs).mean(),
            "nrqm": nrqm_model(sample_outputs).mean(),
            "niqe": niqe_model(sample_outputs).mean(),
        }
    explicit_metrics = {
        name: finite_number(value.item(), f"explicit_validation_{name}")
        for name, value in explicit_metrics.items()
    }

    smoke_dir = HERE / "experiments" / args.run_name
    save_dir = smoke_dir / "checkpoints"
    best_psnr, best_lpips = controlled.save_checkpoint(
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        scheduler_g,
        scheduler_d,
        0,
        float(quick[5]),
        float(quick[7]),
        save_dir,
        float("-inf"),
        float("inf"),
        checkpoint_sha256,
    )
    smoke_checkpoint = save_dir / "controlled_gan_epoch_0.pt"
    saved_sha256 = sha256_file(smoke_checkpoint)

    restored_generator, _, _ = controlled.baseline.create_model(controlled_opt["network"], 0)
    restored_discriminator = controlled.PatchGANDiscriminator(
        input_channels=controlled_opt["discriminator"]["input_channels"],
        base_channels=controlled_opt["discriminator"]["base_channels"],
    ).to(device)
    restored_optimizer_g, restored_scheduler_g = controlled.baseline.create_optim_scheduler(
        controlled_opt["train"], restored_generator
    )
    restored_optimizer_d, restored_scheduler_d = controlled.build_discriminator_optimizer(
        restored_discriminator, controlled_opt["discriminator"]
    )
    restored = torch.load(smoke_checkpoint, map_location=device, weights_only=True)
    restored_generator.load_state_dict(restored["model_state_dict"], strict=True)
    restored_discriminator.load_state_dict(restored["discriminator_state_dict"], strict=True)
    restored_optimizer_g.load_state_dict(restored["optimizer_state_dict"])
    restored_optimizer_d.load_state_dict(restored["optimizer_D_state_dict"])
    restored_scheduler_g.load_state_dict(restored["scheduler_state_dict"])
    restored_scheduler_d.load_state_dict(restored["scheduler_D_state_dict"])

    generator_restore_ok = tensor_equal(
        generator.state_dict()["intro.weight"],
        restored_generator.state_dict()["intro.weight"],
    )
    discriminator_restore_ok = tensor_equal(
        discriminator.state_dict()["model.0.weight"],
        restored_discriminator.state_dict()["model.0.weight"],
    )
    if not generator_restore_ok or not discriminator_restore_ok:
        raise RuntimeError("Checkpoint restore tensor verification failed")

    train_seconds_per_iteration = train_elapsed / args.iterations
    validation_seconds_per_batch = validation_elapsed / len(limited_val)
    train_eta_seconds = (
        train_seconds_per_iteration * args.full_train_batches * controlled.TOTAL_EPOCHS
    )
    quick_val_eta_seconds = (
        validation_seconds_per_batch
        * args.full_quick_validation_batches
        * controlled.TOTAL_EPOCHS
    )
    full_val_eta_seconds = (
        validation_seconds_per_batch * args.full_validation_batches * 10
    )
    total_eta_seconds = train_eta_seconds + quick_val_eta_seconds + full_val_eta_seconds

    result = {
        "status": "PASS",
        "source_commit": "2f08ba5e73347d4e2e3862f81c156ff5089e00a1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "iterations": args.iterations,
        "run_name": args.run_name,
        "validation_batches": len(limited_val),
        "smoke_dataset_root": (
            str(args.dataset_root.resolve()) if args.dataset_root is not None else None
        ),
        "train_batches_per_epoch": len(train_loader),
        "quick_validation_batches": len(val_loader),
        "full_validation_batches": len(full_val_loader),
        "initialization_path": str(checkpoint_path),
        "initialization_sha256": checkpoint_sha256,
        "initialization_epoch": initialization.get("epoch"),
        "generator_lr_after_initial_restore": initial_generator_lrs,
        "generator_scheduler_last_epoch_after_initial_restore": (
            initial_generator_scheduler_last_epoch
        ),
        "generator_scheduler_last_epoch_after_smoke_step": scheduler_g.last_epoch,
        "train_metrics": train_metrics,
        "gradient_checks": {
            "generator": generator_gradients,
            "discriminator": discriminator_gradients,
        },
        "gpu_memory_stability": memory_stability,
        "validation_metrics": {
            "loss": float(quick[0]),
            "psnr": float(quick[5]),
            "ssim": float(quick[6]),
            "lpips_legacy_helper": float(quick[7]),
            "explicit_direct_calls": explicit_metrics,
            "musiq": float(quick[8]),
            "nrqm": float(quick[9]),
            "niqe": float(quick[10]),
        },
        "train_elapsed_seconds": train_elapsed,
        "train_seconds_per_iteration": train_seconds_per_iteration,
        "validation_elapsed_seconds": validation_elapsed,
        "validation_seconds_per_batch": validation_seconds_per_batch,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
        "smoke_checkpoint": str(smoke_checkpoint),
        "smoke_checkpoint_sha256": saved_sha256,
        "generator_restore_ok": generator_restore_ok,
        "discriminator_restore_ok": discriminator_restore_ok,
        "restored_generator_optimizer_state_entries": len(
            restored_optimizer_g.state_dict()["state"]
        ),
        "restored_discriminator_optimizer_state_entries": len(
            restored_optimizer_d.state_dict()["state"]
        ),
        "restored_generator_scheduler_last_epoch": restored_scheduler_g.last_epoch,
        "restored_discriminator_scheduler_last_epoch": restored_scheduler_d.last_epoch,
        "estimated_full_run": {
            "assumed_train_batches_per_epoch": args.full_train_batches,
            "assumed_quick_validation_batches_per_epoch": (
                args.full_quick_validation_batches
            ),
            "assumed_full_validation_batches_per_run": args.full_validation_batches,
            "training_seconds": train_eta_seconds,
            "quick_validation_seconds": quick_val_eta_seconds,
            "full_validation_seconds": full_val_eta_seconds,
            "total_seconds": total_eta_seconds,
            "total_hours": total_eta_seconds / 3600.0,
            "note": "Linear smoke-test estimate; excludes startup, logging, image writes, and variance.",
        },
        "best_psnr_smoke": best_psnr,
        "best_lpips_smoke": best_lpips,
    }
    smoke_dir.mkdir(parents=True, exist_ok=True)
    result_path = smoke_dir / "smoke_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"SMOKE_TEST_PASS result={result_path}")


if __name__ == "__main__":
    main()
