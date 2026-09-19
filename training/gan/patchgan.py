"""PatchGAN discriminator used only by the controlled adversarial branch."""

import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """Five-layer PatchGAN matching the discriminator topology in training/gan."""

    def __init__(self, input_channels=3, base_channels=64):
        super().__init__()
        c = base_channels
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, image):
        return self.model(image)
