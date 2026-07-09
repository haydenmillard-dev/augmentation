import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = nn.Sequential(
                nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
        )
        # self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.convt(x)
        x = F.interpolate(x, size=skip_connection.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip_connection, x], dim=1)
        x = self.double_conv(x)
        return x
