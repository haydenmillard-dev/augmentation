import torch
import torch.nn as nn
import torchvision

class DeepLabV3PlusRGB(nn.Module):
    """
    Standard DeepLabV3+ model for 3-channel RGB input.
    Uses pretrained ResNet-101 backbone.
    """

    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()

        # -----------------------------
        # 1. Load pretrained DeepLabV3+
        # -----------------------------
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DEFAULT" if pretrained else None,
            weights_backbone="DEFAULT" if pretrained else None
        )

        # -----------------------------
        # 2. Replace classifier head
        # -----------------------------
        # Original head outputs 21 classes (VOC default)
        in_channels = self.model.classifier[-1].in_channels

        self.model.classifier[-1] = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1
        )

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, x):
        """
        Expects input shape:
        [B, 3, H, W]
        """
        return self.model(x)

class DeepLabV3PlusMultiChannel(nn.Module):
    """
    DeepLabV3+ wrapper that allows N-channel input instead of RGB
    while preserving pretrained weights.
    """

    def __init__(self, num_input_channels=3, num_classes=21, pretrained=True):
        super().__init__()

        # -----------------------------
        # 1. Load pretrained DeepLabV3+
        # -----------------------------
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DEFAULT" if pretrained else None,
            weights_backbone="DEFAULT" if pretrained else None
        )

        # Replace classifier head if needed
        self.model.classifier[-1] = nn.Conv2d(
            256, num_classes, kernel_size=1
        )

        # -----------------------------
        # 2. Modify first conv layer
        # -----------------------------
        self._adapt_first_conv(num_input_channels)

    # -------------------------------------------------
    # Core function: modify first conv layer safely
    # -------------------------------------------------
    def _adapt_first_conv(self, in_channels):
        """
        Expands first conv layer from 3 → N channels
        while preserving pretrained RGB weights.
        """

        backbone = self.model.backbone

        # ResNet first conv layer
        old_conv = backbone.conv1

        if in_channels == 3:
            return  # nothing to change

        # Create new conv
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # -----------------------------
        # Copy pretrained weights
        # -----------------------------
        with torch.no_grad():

            # 1. Copy RGB weights directly
            new_conv.weight[:, :3] = old_conv.weight

            # 2. Initialize extra channels safely
            if in_channels > 3:
                rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)

                new_conv.weight[:, 3:] = rgb_mean

        # Replace conv layer in backbone
        backbone.conv1 = new_conv

    # -------------------------------------------------
    # Forward pass
    # -------------------------------------------------
    def forward(self, x):
        return self.model(x)
