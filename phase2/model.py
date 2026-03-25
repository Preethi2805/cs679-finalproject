import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> InstanceNorm2d -> ReLU"""

    def __init__(self, in_c, out_c, kernel, stride=1, padding=0, norm=True, relu=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding,
                      padding_mode='reflect')
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_c, affine=True))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """
    Residual block — maintains spatial resolution.
    Two conv layers with instance norm; input is added back (skip connection).
    """

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class TransformNet(nn.Module):
    """
    Feed-forward style transfer network (Johnson et al. 2016).

    Architecture:
        Encoder  : 3 conv layers, downsamples 4x
        Residuals: 5 residual blocks, maintains resolution
        Decoder  : 2 upsample + conv layers, restores original resolution

    Input:  (B, 3, H, W) — normalised to [0, 1]
    Output: (B, 3, H, W) — pixel values in [0, 1]
    """

    def __init__(self, num_res_blocks=5):
        super().__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            ConvBlock(3,   32,  9, stride=1, padding=4),   # same resolution
            ConvBlock(32,  64,  3, stride=2, padding=1),   # /2
            ConvBlock(64,  128, 3, stride=2, padding=1),   # /4
        )

        # --- Residual blocks ---
        self.residuals = nn.Sequential(
            *[ResBlock(128) for _ in range(num_res_blocks)]
        )

        # --- Decoder ---
        # Upsample with nearest-neighbour then conv (avoids checkerboard artifacts
        # that transposed convolutions produce)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(128, 64, 3, padding=1),              # x2
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(64,  32, 3, padding=1),              # x4
            # Final layer: no norm, no relu — full range output
            nn.Conv2d(32, 3, 9, padding=4, padding_mode='reflect'),
            nn.Sigmoid(),                                  # clamp to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    net = TransformNet()
    dummy = torch.randn(2, 3, 256, 256)
    out = net(dummy)
    print(f'Input:  {dummy.shape}')
    print(f'Output: {out.shape}')
    total = sum(p.numel() for p in net.parameters())
    print(f'Parameters: {total:,}')