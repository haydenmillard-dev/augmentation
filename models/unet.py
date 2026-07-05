import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sequential = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.sequential(x)

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

class InputAdapter(nn.Module):
    """
    A single layer adapter that combines M input channels into N output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

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

ADAPTERS = [None, "input", "multi-layer", "deep-input"]

class Encoder(nn.Module):
    def __init__(self, backbone, in_channels=3):
        super().__init__()
        self.backbone = backbone

        if backbone == "resnet":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
            self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
            self.encoder2 = nn.Sequential(resnet.layer2)
            self.encoder3 = nn.Sequential(resnet.layer3)
            self.encoder4 = nn.Sequential(resnet.layer4)
        elif backbone is None:
            self.encoder0 = DoubleConv(in_channels, 64)
            self.encoder1 = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(64, 128))
            self.encoder2 = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(128, 256))
            self.encoder3 = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(256, 512))
            self.encoder4 = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(512, 1024), nn.Dropout2d())
        else:
            raise ValueError(f"Invalid option given for backbone. Model received {backbone}.")

    def forward(self, x):
        skip3 = self.encoder0(x)
        skip2 = self.encoder1(skip3)
        skip1 = self.encoder2(skip2)
        skip0 = self.encoder3(skip1)
        x = self.encoder4(skip0)

        return x, skip0, skip1, skip2, skip3

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder4 = UpsamplingBlock(512, 256, 256)
        self.decoder3 = UpsamplingBlock(256, 128, 128)
        self.decoder2 = UpsamplingBlock(128, 64, 64)
        self.decoder1 = UpsamplingBlock(64, 64, 64)
    
    def forward(self, x, skip0, skip1, skip2, skip3):
        x = self.decoder4(x, skip0)
        x = self.decoder3(x, skip1)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip3)
        return x

class UNet(nn.Module):
    def __init__(self, adapter, backbone, in_channels, intermediate_channels=3, hidden_channels=32, num_classes=21, drop_probability=0.25):
        super().__init__()

        if adapter == "none":
            self.adapter = None
        elif adapter == "input":
            self.adapter = InputAdapter(in_channels, intermediate_channels)
        elif adapter == "multi-layer":
            self.adapter = MultiLayerAdapter()
        elif adapter == "deep-input":
            self.adapter = DeepInputAdapter(in_channels, intermediate_channels, hidden_channels, drop_probability)

        self.encoder = Encoder(backbone, in_channels)
        self.decoder = Decoder()
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        if self.adapter:
            x = self.adapter(x)
        x, skip1, skip2, skip3, skip4 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, skip4)
        return self.final(x)

    def get_module_params(self):
        '''Returns the parameters for the adapter (if it exists), encoder and decoder separately'''
        params = []
        if self.adapter:
            params.append(list(self.adapter.parameters()))
        params.append(list(self.encoder.parameters()))
        params.append(list(self.decoder.parameters()))
        params.append(list(self.final.parameters()))
        return params
