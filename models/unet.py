import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Contracting path
        self.downconv1 = DoubleConv(3, 64)
        self.max1 = nn.MaxPool2d(2, 2)

        self.downconv2 = DoubleConv(64, 128)
        self.max2 = nn.MaxPool2d(2, 2)

        self.downconv3 = DoubleConv(128, 256)
        self.max3 = nn.MaxPool2d(2, 2)

        self.downconv4 = DoubleConv(256, 512)
        self.max4 = nn.MaxPool2d(2, 2)

        # Trough
        self.downconv5 = DoubleConv(512, 1024)

        # Expanding path
        self.transpose1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)

        self.upconv1 = DoubleConv(1024, 512)
        self.transpose2 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.upconv2 = DoubleConv(512, 256)
        self.transpose3 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.upconv3 = DoubleConv(256, 128)
        self.transpose4 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.upconv4 = DoubleConv(128, 64)
        self.finalconv = nn.Conv2d(64, num_classes, 1)


    def forward(self, X):
        # Contracting path
        skip_connection_1 = self.downconv1(X)
        X = self.max1(skip_connection_1)

        skip_connection_2 = self.downconv2(X)
        X = self.max2(skip_connection_2)

        skip_connection_3 = self.downconv3(X)
        X = self.max3(skip_connection_3)

        skip_connection_4 = self.downconv4(X)
        X = self.max4(skip_connection_4)

        # Trough
        X = self.downconv5(X)

        # Expanding path
        X = self.transpose1(X)
        X = torch.concat((skip_connection_4, X), dim=1)

        X = self.upconv1(X)
        X = self.transpose2(X)
        X = torch.concat((skip_connection_3, X), dim=1)

        X = self.upconv2(X)
        X = self.transpose3(X)
        X = torch.concat((skip_connection_2, X), dim=1)

        X = self.upconv3(X)
        X = self.transpose4(X)
        X = torch.concat((skip_connection_1, X), dim=1)

        X = self.upconv4(X)
        X = self.finalconv(X)
        
        return X

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sequential = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
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
        )
        # self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = F.interpolate(self.convt(x), size=skip_connection.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip_connection, x], dim=1)
        x = self.double_conv(x)
        return x

class UNetResNet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        if in_channels > 3:
            self.adapter = InputAdapter(in_channels, 3)
        # Contracting path
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2 = nn.Sequential(resnet.layer2)
        self.encoder3 = nn.Sequential(resnet.layer3)
        self.encoder4 = nn.Sequential(resnet.layer4)
        
        # Expanding path
        self.decoder4 = UpsamplingBlock(512, 256, 256)
        self.decoder3 = UpsamplingBlock(256, 128, 128)
        self.decoder2 = UpsamplingBlock(128, 64, 64)
        self.decoder1 = UpsamplingBlock(64, 64, 64)

        self.final = nn.Sequential(
                nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2),
                nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Contracting path
        if x.shape[1] > 3:
            x = self.adapter(x)
        skip0 = self.encoder0(x)
        skip1 = self.encoder1(skip0)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)

        # Expanding path
        x = self.decoder4(skip4, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder2(x, skip1)
        x = self.decoder1(x, skip0)

        x = self.final(x)

        return x

class InputAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DeepInputAdapter(nn.Module):
    def __init__(self, in_channels=7, out_channels=3, hidden_channels=32, dropout=0.1):
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

            nn.Dropout2d(dropout),

            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)

class UNetResNetAdapter(nn.Module):
    def __init__(self, num_classes, in_channels=3, adapter="multi-layer"):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Adapter
        if adapter == "multi-layer":
            self.map = True
            self.adapter = MultiLayerAdapter()
        elif adapter == "deep-input":
            self.map = True
            self.adapter = DeepInputAdapter(in_channels=in_channels, out_channels=3, hidden_channels=32, dropout=0.1)

        # Contracting path
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2 = nn.Sequential(resnet.layer2)
        self.encoder3 = nn.Sequential(resnet.layer3)
        self.encoder4 = nn.Sequential(resnet.layer4)
        
        # Expanding path
        self.decoder4 = UpsamplingBlock(512, 256, 256)
        self.decoder3 = UpsamplingBlock(256, 128, 128)
        self.decoder2 = UpsamplingBlock(128, 64, 64)
        self.decoder1 = UpsamplingBlock(64, 64, 64)

        self.final = nn.Sequential(
                nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2),
                nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Adapter
        x = self.adapter(x)
        
        # Contracting path
        skip0 = self.encoder0(x)
        skip1 = self.encoder1(skip0)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)

        # Expanding path
        x = self.decoder4(skip4, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder2(x, skip1)
        x = self.decoder1(x, skip0)

        x = self.final(x)

        return x

class MultiLayerAdapter(nn.Module):
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
