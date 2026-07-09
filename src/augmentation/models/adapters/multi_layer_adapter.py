import torch
import torch.nn as nn

class MultiLayerAdapter(nn.Module):
    """
    A multi-layer adapter that first combines related channels (such as RGB) and then combines these representations at the bottleneck.
    """
    def __init__(self):
        super().__init__()
        self.rgb_adapter = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.posterised_adapter = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.edge_adapter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.bottleneck = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)

    def forward(self, x):
        x1 = self.rgb_adapter(x[:, 0:3])
        x2 = self.posterised_adapter(x[:, 3:6])
        x3 = self.edge_adapter(x[:, 6:7])
        
        return self.bottleneck(torch.cat([x1, x2, x3], dim=1))

