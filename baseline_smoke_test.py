"""Bounded smoke test for the baseline_continued control run."""

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
BASELINE_DIR = HERE / "training" / "baseline_continued"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import train_continued as baseline  # noqa: E402


TOTAL_EPOCHS = 30
LEARNING_RATE = 1e-4
ETA_MIN = 1e-6


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
                    "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
                    "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
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


def gradient_summary(module):
    gradients = [
        parameter.grad.detach()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("generator has no gradients")
    finite = all(torch.isfinite(gradient).all().item() for gradient in gradients)
    squared_norm = sum(
        float(torch.sum(gradient.float() * gradient.float()).item())
        for gradient in gradients
    )
    norm = math.sqrt(squared_norm)
    nonzero_tensors = sum(
        int(torch.count_nonzero(gradient).item() > 0) for gradient in gradients
    )
    if not finite or not math.isfinite(norm) or norm <= 0 or nonzero_tensors == 0:
        raise RuntimeError(
            "generator gradients invalid: "
            f"finite={finite}, norm={norm}, nonzero_tensors={nonzero_tensors}"
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
    parser.add_argument("--run-name", default="smoke_strict")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 100 <= args.iterations <= 500:
        raise ValueError("--iterations must be between 100 and 500")
    if args.validation_batches < 1:
        raise ValueError("--validation-batches must be positive")

    os.chdir(BASELINE_DIR)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    opt = baseline.load_config("options/config.yaml")
    opt["train"]["epochs"] = TOTAL_EPOCHS
    opt["train"]["lr_initial"] = LEARNING_RATE
    opt["train"]["eta_min"] = ETA_MIN

    model, _, _ = baseline.create_model(opt["network"], 0)
    criterion = baseline.create_loss(opt["train"]["loss"], rank=0)
    optimizer, scheduler = baseline.create_optim_scheduler(opt["train"], model)

    checkpoint_path = (BASELINE_DIR / opt["path"]["resume_path"]).resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"initialization checkpoint missing: {missing}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if not torch.equal(
        model.state_dict()["intro.weight"].detach().cpu(),
        checkpoint["model_state_dict"]["intro.weight"].detach().cpu(),
    ):
        raise RuntimeError("initialization tensor verification failed")
    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    initial_scheduler_epoch = scheduler.last_epoch

    train_loader, train_sampler = baseline.create_dataloaders(
        0, 1, opt["datasets"], mode="train", return_path=False, samples_per_scene=15
    )
    val_loader, _ = baseline.create_dataloaders(
        0, 1, opt["datasets"], mode="val", return_path=False, samples_per_scene=15
    )
    full_val_loader, _ = baseline.create_dataloaders(
        0, 1, opt["datasets"], mode="val", return_path=False, samples_per_scene=-1
    )
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(0)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    monitored_train = LimitedLoader(train_loader, args.iterations)
    train_start = time.perf_counter()
    train_metrics_tuple = baseline.train_one_epoch(
        model, monitored_train, criterion, optimizer, device
    )
    torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - train_start
    scheduler.step()
    gradient_checks = gradient_summary(model)

    metric_names = ("total", "pixel", "perceptual", "edge", "enhance")
    train_metrics = {
        name: finite_number(value, f"train_{name}")
        for name, value in zip(metric_names, train_metrics_tuple)
    }
    if train_metrics["perceptual"] <= 0:
        raise RuntimeError("LPIPS/perceptual loss did not execute")

    memory_samples = monitored_train.memory_samples
    if len(memory_samples) != args.iterations:
        raise RuntimeError("GPU memory sampling incomplete")
    warmup_start = min(9, len(memory_samples) - 1)
    post_warmup = memory_samples[warmup_start:]
    allocated = [sample["allocated_gib"] for sample in post_warmup]
    reserved = [sample["reserved_gib"] for sample in post_warmup]
    memory_stability = {
        "warmup_ignored_iterations": warmup_start,
        "post_warmup_allocated_span_gib": max(allocated) - min(allocated),
        "post_warmup_reserved_span_gib": max(reserved) - min(reserved),
        "post_warmup_allocated_end_minus_start_gib": allocated[-1] - allocated[0],
        "post_warmup_reserved_end_minus_start_gib": reserved[-1] - reserved[0],
    }
    memory_stability["stable"] = (
        abs(memory_stability["post_warmup_allocated_end_minus_start_gib"]) < 0.5
        and abs(memory_stability["post_warmup_reserved_end_minus_start_gib"]) < 0.5
        and memory_stability["post_warmup_allocated_span_gib"] < 1.0
        and memory_stability["post_warmup_reserved_span_gib"] < 1.0
    )
    if not memory_stability["stable"]:
        raise RuntimeError(f"GPU memory did not stabilize: {memory_stability}")

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
    psnr_model = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_model = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    limited_val = LimitedLoader(val_loader, args.validation_batches)
    torch.cuda.synchronize()
    validation_start = time.perf_counter()
    quick = baseline.validate_quick(
        model,
        limited_val,
        criterion,
        device,
        psnr_model,
        ssim_model,
        lpips_model,
        musiq_model,
        nrqm_model,
        niqe_model,
        0,
        desc="Baseline smoke validation",
    )
    torch.cuda.synchronize()
    validation_elapsed = time.perf_counter() - validation_start
    for index, value in enumerate(quick):
        finite_number(value, f"validation_output_{index}")

    sample_inputs, sample_targets = next(iter(val_loader))
    with torch.no_grad():
        sample_outputs = torch.clamp(model(sample_inputs.to(device)), 0.0, 1.0)
        sample_targets = sample_targets.to(device)
        direct_metrics = {
            "psnr": psnr_model(sample_outputs, sample_targets),
            "ssim": ssim_model(sample_outputs, sample_targets),
            "lpips": lpips_model(sample_outputs, sample_targets).mean(),
            "musiq": musiq_model(sample_outputs).mean(),
            "nrqm": nrqm_model(sample_outputs).mean(),
            "niqe": niqe_model(sample_outputs).mean(),
        }
    direct_metrics = {
        name: finite_number(value.item(), f"direct_{name}")
        for name, value in direct_metrics.items()
    }

    smoke_dir = BASELINE_DIR / "experiments" / args.run_name
    save_dir = smoke_dir / "checkpoints"
    baseline.save_checkpoint_baseline(
        model,
        optimizer,
        scheduler,
        0,
        float(quick[5]),
        float(quick[7]),
        str(save_dir),
        float("-inf"),
        float("inf"),
        is_periodic=True,
    )
    smoke_checkpoint = save_dir / "baseline_epoch_0.pt"
    smoke_checkpoint_sha256 = sha256_file(smoke_checkpoint)

    restored_model, _, _ = baseline.create_model(opt["network"], 0)
    restored_optimizer, restored_scheduler = baseline.create_optim_scheduler(
        opt["train"], restored_model
    )
    restored = torch.load(smoke_checkpoint, map_location=device, weights_only=False)
    restored_model.load_state_dict(restored["model_state_dict"], strict=True)
    restored_optimizer.load_state_dict(restored["optimizer_state_dict"])
    restored_scheduler.load_state_dict(restored["scheduler_state_dict"])
    restore_ok = torch.equal(
        model.state_dict()["intro.weight"].detach().cpu(),
        restored_model.state_dict()["intro.weight"].detach().cpu(),
    )
    if not restore_ok:
        raise RuntimeError("checkpoint restore tensor verification failed")

    train_seconds_per_iteration = train_elapsed / args.iterations
    validation_seconds_per_batch = validation_elapsed / len(limited_val)
    total_eta_seconds = (
        train_seconds_per_iteration * len(train_loader) * TOTAL_EPOCHS
        + validation_seconds_per_batch * len(val_loader) * TOTAL_EPOCHS
        + validation_seconds_per_batch * len(full_val_loader) * 10
    )

    result = {
        "status": "PASS",
        "source_commit": "2f08ba5e73347d4e2e3862f81c156ff5089e00a1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "iterations": args.iterations,
        "train_batches_per_epoch": len(train_loader),
        "quick_validation_batches": len(val_loader),
        "full_validation_batches": len(full_val_loader),
        "initialization_path": str(checkpoint_path),
        "initialization_sha256": checkpoint_sha256,
        "initialization_epoch": checkpoint.get("epoch"),
        "generator_lr_after_restore": initial_lrs,
        "scheduler_last_epoch_after_restore": initial_scheduler_epoch,
        "scheduler_last_epoch_after_smoke_step": scheduler.last_epoch,
        "train_metrics": train_metrics,
        "gradient_checks": gradient_checks,
        "gpu_memory_stability": memory_stability,
        "validation_metrics": {
            "loss": float(quick[0]),
            "psnr": float(quick[5]),
            "ssim": float(quick[6]),
            "lpips": float(quick[7]),
            "musiq": float(quick[8]),
            "nrqm": float(quick[9]),
            "niqe": float(quick[10]),
            "explicit_direct_calls": direct_metrics,
        },
        "train_elapsed_seconds": train_elapsed,
        "train_seconds_per_iteration": train_seconds_per_iteration,
        "validation_elapsed_seconds": validation_elapsed,
        "validation_seconds_per_batch": validation_seconds_per_batch,
        "estimated_full_run_seconds": total_eta_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "smoke_checkpoint": str(smoke_checkpoint),
        "smoke_checkpoint_sha256": smoke_checkpoint_sha256,
        "checkpoint_restore_ok": restore_ok,
        "restored_optimizer_state_entries": len(
            restored_optimizer.state_dict()["state"]
        ),
    }
    smoke_dir.mkdir(parents=True, exist_ok=True)
    result_path = smoke_dir / "smoke_result.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"BASELINE_SMOKE_PASS: {result_path}")


if __name__ == "__main__":
    main()
