#!/usr/bin/env python3
"""Static pre-publication checks that require only the Python standard library."""

import ast
import hashlib
import json
from pathlib import Path

from summarize_stage3 import verify_and_summarize


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = json.loads(
    (ROOT / "results" / "stage3" / "provenance.json").read_text(encoding="utf-8")
)
REQUIRED = (
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    "docs/reproduction.md",
    "docs/implementation_notes.md",
    "docs/model_assets.md",
    "results/stage3/baseline_continued.csv",
    "results/stage3/darkir_gan.csv",
    "results/stage3/provenance.json",
    "training/baseline_continued/train_continued.py",
    "training/gan/README.md",
    "training/gan/options/config.yaml",
    "training/gan/patchgan.py",
    "training/gan/train_controlled_gan.py",
    "training/gan/verify_setup.py",
)
BINARY_SUFFIXES = {".pt", ".pth", ".ckpt"}
FORBIDDEN_LEGACY_PATHS = (
    "training/gan_controlled",
    "training/gan/archs",
    "training/gan/data",
    "training/gan/losses",
    "training/gan/train_gan.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"Missing release files: {missing}")

    retained_legacy = [
        path for path in FORBIDDEN_LEGACY_PATHS if (ROOT / path).exists()
    ]
    if retained_legacy:
        raise RuntimeError(f"Historical GAN paths remain in release: {retained_legacy}")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if not license_text.startswith("MIT License\n"):
        raise RuntimeError("LICENSE is not the canonical MIT license text")
    if "Copyright (c) 2026 Xinyu He" not in license_text:
        raise RuntimeError("Project copyright notice is missing from LICENSE")

    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    if "Copyright (c) 2025 cidautai" not in notices:
        raise RuntimeError("The upstream DarkIR copyright notice is missing")
    if "https://github.com/cidautai/DarkIR" not in notices:
        raise RuntimeError("The upstream DarkIR source URL is missing")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "[MIT License](LICENSE)" not in readme:
        raise RuntimeError("README does not identify the project license")
    if "docs/model_assets.md" not in readme:
        raise RuntimeError("README does not link the separate asset manifest")
    release_url = "https://github.com/David-He-NJ/DarkIR-GAN/releases/tag/v1.0.0"
    if release_url not in readme:
        raise RuntimeError("README does not link the v1.0.0 model release")
    if PROVENANCE.get("model_release", {}).get("url") != release_url:
        raise RuntimeError("Provenance does not identify the v1.0.0 model release")

    for relative_path, expected in PROVENANCE["source_files"].items():
        path = ROOT / relative_path
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"AutoDL source hash mismatch: {relative_path}: {actual}")

    unwanted = []
    oversized = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative_parts = path.relative_to(ROOT).parts
        if relative_parts[0] in {"datasets", "weights"} or "experiments" in relative_parts:
            continue
        if path.name == ".DS_Store" or path.suffix.lower() in BINARY_SUFFIXES:
            unwanted.append(path.relative_to(ROOT).as_posix())
        if path.stat().st_size > 10 * 1024 * 1024:
            oversized.append(path.relative_to(ROOT).as_posix())
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    if unwanted:
        raise RuntimeError(f"Release contains excluded binary/cache files: {unwanted}")
    if oversized:
        raise RuntimeError(f"Release contains files larger than 10 MiB: {oversized}")

    verify_and_summarize(check=True)
    print("PUBLIC_RELEASE_STATIC_CHECK_OK")


if __name__ == "__main__":
    main()
