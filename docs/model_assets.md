# Separately hosted model assets

Model checkpoints are distributed separately from the Git repository. Keeping
the binaries outside Git makes the source history reviewable and avoids
duplicating multi-gigabyte training artifacts. Official download links will be
added to the main README when the external release is uploaded.

Each published file must be verified against the SHA-256 value below before it
is used. The suggested release filenames are unique so that the Stage II
initialization cannot be confused with the Stage III best-PSNR checkpoint.

| Artifact | Suggested release filename | Original/runtime filename | SHA-256 |
|---|---|---|---|
| Shared Stage II initialization | `darkir_stage2_init_baseline_best_psnr.pt` | `training/baseline/experiments/checkpoints/baseline_best_psnr.pt` | `dbfdc6494daae9a809607f63e74a6193a8b4ed0004c02ae9f23dda1cfc723a99` |
| Stage III baseline, epoch 30 | `darkir_stage3_baseline_epoch_30.pt` | `baseline_epoch_30.pt` | `a5f01db6e525b1a15f54a1369d77da15ac89c8083fb7094246aebe1d554f9620` |
| Stage III baseline, best PSNR | `darkir_stage3_baseline_best_psnr.pt` | `baseline_best_psnr.pt` | `844edbdbaad745c27d7b82e9dbcef322c352d0b02cfeb9518cb828cc34b321` |
| Stage III baseline, best LPIPS | `darkir_stage3_baseline_best_lpips.pt` | `baseline_best_lpips.pt` | `e14d1924739968fe221c8b800fe6f4aa0c046bfb8265f7d0753e5e93127d9b19` |
| Stage III DarkIR-GAN, epoch 30 | `darkir_gan_stage3_epoch_30.pt` | `controlled_gan_epoch_30.pt` | `c3e7d276149a0935b9d92e0466aad60848767f04cc49023e90bc08aa585614bd` |
| Stage III DarkIR-GAN, best PSNR | `darkir_gan_stage3_best_psnr.pt` | `controlled_gan_best_psnr.pt` | `0789597348595bb77a704ec4d7700ef4c6be86bb09bd463fa28373d9003897a6` |
| Stage III DarkIR-GAN, best LPIPS | `darkir_gan_stage3_best_lpips.pt` | `controlled_gan_best_lpips.pt` | `4811df11ded9fc0358fb8b48962844e83780c51306c2bda92710b39779532cf1` |

For the Stage III launch scripts, download the shared initialization, rename it
to `baseline_best_psnr.pt`, and place it at the runtime path shown above. The
other six checkpoints are release/evaluation artifacts and are not required to
start the matched Stage III training runs.

Recommended publication procedure:

1. Upload the seven files to a versioned external release with stable URLs.
2. Verify every remote download with `shasum -a 256 <filename>`.
3. Add the download URL and archive version to the main README.
4. Keep all `.pt`, `.pth` and `.ckpt` files outside Git; the release checker
   rejects checkpoint binaries in the source package.
