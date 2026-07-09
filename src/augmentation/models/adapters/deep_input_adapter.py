import torch.nn as nn

class DeepInputAdapter(nn.Module):
    """
    A multi-layer adapter that combines M channels over multiple layers of H hidden channels and outputs N channels.
    """
    def __init__(self, in_channels=7, out_channels=3, hidden_channels=32, drop_probability=0.25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(drop_probability),

            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)

