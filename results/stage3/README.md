# Stage III full-validation results

These CSVs are the complete scheduled validation records from the two completed 30-epoch AutoDL runs:

- `baseline_continued.csv`: reconstruction-only continuation.
- `darkir_gan.csv`: the matched continuation plus PatchGAN/LSGAN supervision.

Each file contains evaluations at epochs 3, 6, 9, ..., 30 on all 1,800 validation images. `provenance.json` records source, initialization, CSV, smoke-result and checkpoint hashes.

Verify and summarize the files with:

```bash
python scripts/summarize_stage3.py --check
```

All ten validation points favor DarkIR-GAN on LPIPS, MUSIQ, NIQE and NRQM. This is descriptive evidence from one run per branch, not a repeated-seed significance test.

The checkpoint binaries and 160 saved visualization images are not committed. Their hashes and counts are retained in `provenance.json`; checkpoint release filenames and the separate-hosting policy are documented in [`../../docs/model_assets.md`](../../docs/model_assets.md).
