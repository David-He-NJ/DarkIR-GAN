#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
checkpoint="$repo_root/training/baseline/experiments/checkpoints/baseline_best_psnr.pt"
expected_checkpoint_sha="dbfdc6494daae9a809607f63e74a6193a8b4ed0004c02ae9f23dda1cfc723a99"

if [[ $# -ne 1 || ( "$1" != "baseline" && "$1" != "gan" ) ]]; then
  echo "Usage: $0 baseline|gan" >&2
  exit 64
fi

if [[ ! -f "$checkpoint" ]]; then
  echo "Missing shared Stage II checkpoint: $checkpoint" >&2
  exit 66
fi

actual_checkpoint_sha="$($python_bin -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$checkpoint")"
if [[ "$actual_checkpoint_sha" != "$expected_checkpoint_sha" ]]; then
  echo "Initialization SHA-256 mismatch: $actual_checkpoint_sha" >&2
  exit 65
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

if [[ "$1" == "baseline" ]]; then
  run_dir="$repo_root/training/baseline_continued"
  entry="train_continued.py"
else
  "$python_bin" "$repo_root/training/gan/verify_setup.py"
  run_dir="$repo_root/training/gan"
  entry="train_controlled_gan.py"
fi

if [[ -d "$run_dir/experiments" && "${ALLOW_EXISTING_OUTPUTS:-0}" != "1" ]]; then
  if find "$run_dir/experiments" -type f -print -quit | grep -q .; then
    echo "Refusing to mix with existing outputs under $run_dir/experiments" >&2
    echo "Move them away or set ALLOW_EXISTING_OUTPUTS=1 deliberately." >&2
    exit 70
  fi
fi

cd "$run_dir"
exec "$python_bin" -u "$entry"
