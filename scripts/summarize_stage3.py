#!/usr/bin/env python3
"""Verify and summarize the published Stage III validation trajectories."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = ROOT / "results" / "stage3" / "provenance.json"
METRICS = (
    ("val_psnr", "PSNR", True),
    ("val_ssim", "SSIM", True),
    ("val_lpips", "LPIPS", False),
    ("val_musiq", "MUSIQ", True),
    ("val_niqe", "NIQE", False),
    ("val_nrqm", "NRQM", True),
)
EXPECTED_EPOCHS = list(range(3, 31, 3))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {key: int(value) if key == "epoch" else float(value) for key, value in row.items()}
        for row in rows
    ]


def verify_and_summarize(check=False):
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    runs = {}
    for name, record in provenance["runs"].items():
        csv_path = ROOT / record["csv"]
        actual_hash = sha256_file(csv_path)
        if actual_hash != record["csv_sha256"]:
            raise RuntimeError(f"CSV hash mismatch for {name}: {actual_hash}")
        rows = load_rows(csv_path)
        epochs = [row["epoch"] for row in rows]
        if epochs != EXPECTED_EPOCHS:
            raise RuntimeError(f"Unexpected epoch schedule for {name}: {epochs}")
        runs[name] = rows

    if check:
        for name, expected in provenance["reported_final"].items():
            final = runs[name][-1]
            for metric, value in expected.items():
                if abs(final[metric] - value) > 1e-12:
                    raise RuntimeError(
                        f"Reported endpoint mismatch: {name} {metric} "
                        f"{final[metric]} != {value}"
                    )

    baseline = runs["baseline_continued"]
    gan = runs["darkir_gan"]
    final_baseline = baseline[-1]
    final_gan = gan[-1]

    print("| Method | PSNR | SSIM | LPIPS | MUSIQ | NIQE | NRQM |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, row in (("Baseline-continued", final_baseline), ("DarkIR-GAN", final_gan)):
        values = [row[key] for key, _, _ in METRICS]
        print(
            f"| {label} | {values[0]:.3f} | {values[1]:.4f} | "
            f"{values[2]:.4f} | {values[3]:.3f} | {values[4]:.3f} | "
            f"{values[5]:.3f} |"
        )

    print("\nEndpoint delta (DarkIR-GAN - baseline):")
    for key, label, _ in METRICS:
        print(f"  {label}: {final_gan[key] - final_baseline[key]:+.6f}")

    print("\nValidation points favoring DarkIR-GAN:")
    for key, label, higher_is_better in METRICS:
        count = sum(
            (gan_row[key] > base_row[key])
            if higher_is_better
            else (gan_row[key] < base_row[key])
            for base_row, gan_row in zip(baseline, gan)
        )
        print(f"  {label}: {count}/{len(gan)}")

    if check:
        print("\nSTAGE3_RESULTS_CHECK_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="enforce published values")
    args = parser.parse_args()
    verify_and_summarize(check=args.check)


if __name__ == "__main__":
    main()
