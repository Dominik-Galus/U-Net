import torch
from torch import nn

from u_net.building_blocks import DownBlock, UpBlock


class UNet(nn.Module):
    def __init__(self, channels: int, num_outputs: int) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList([
            DownBlock(channels, 64),
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 512),
        ])
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3),
            nn.ReLU(),
        )
        self.len_dblocks = len(self.down_blocks)

        self.up_blocks = nn.ModuleList([
            UpBlock(in_channels=1024, out_channels=512),
            UpBlock(in_channels=512, out_channels=256),
            UpBlock(in_channels=256, out_channels=128),
            UpBlock(in_channels=128, out_channels=64),
        ])

        self.out_conv = nn.Conv2d(
            in_channels=64, out_channels=num_outputs, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dblocks_result = []
        for block in self.down_blocks:
            x, to_copy = block(x)
            dblocks_result.append(to_copy)
        x = self.down_conv(x)

        for i, block in enumerate(self.up_blocks):
            x = block(x, dblocks_result[self.len_dblocks - i])
        x = self.out_conv(x)
        return x
