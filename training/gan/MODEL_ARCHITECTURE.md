# DarkIR-GAN architecture

## Generator

`archs/darkir.py` uses 64 initial channels, encoder block counts [2,2,4], decoder counts [2,2,2], and one encoder/decoder bottleneck block. Three stride-2 convolutions increase channels to 512. PixelShuffle stages recover resolution, with additive feature skips and an image-level residual. Decoder dilations are [1,2,3].

Inputs are padded to multiples of eight, then cropped back. The architecture is not restricted to 256×256 inputs. The optional side-output branch exists, but the GAN wrapper does not enable side supervision.

## Discriminator

There are **five** 4×4 convolutions:

| Layer | Channels | Stride | Following operations |
|---|---|---|---|
| 1 | 3 → 64 | 2 | LeakyReLU(0.2) |
| 2 | 64 → 128 | 2 | BatchNorm, LeakyReLU(0.2) |
| 3 | 128 → 256 | 2 | BatchNorm, LeakyReLU(0.2) |
| 4 | 256 → 512 | 1 | BatchNorm, LeakyReLU(0.2) |
| 5 | 512 → 1 | 1 | None |

The nominal convolutional receptive field is 70×70. Outputs are unconstrained scores, not sigmoid probabilities. Targets and generated images are processed separately. The discriminator is required only during training.

## Loss accounting

The gradient-bearing GAN objective is:

`L_G = L_pixel + L_perceptual + 0.01 L_laplacian + 0.01 L_adversarial`.

Terms here are unweighted before the displayed coefficient. The edge coefficient includes multiplication inside the loss and in the loop. LSGAN uses discriminator targets 0.9 (real) and 0 (fake), and generator target 1.

The historical `enhance_loss(inputs, targets)` is input–target L1; it has zero generator gradient. It is not intermediate-feature supervision. Original code/log accounting is retained.

The standard backbone of `lpips.LPIPS(net='vgg')` is VGG16, not the VGG19 branch also defined in the file. Its historical call does not explicitly enable LPIPS input conversion. Baseline enhancement, clipping and weights differ. See [implementation notes](../../docs/implementation_notes.md).
