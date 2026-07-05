import torch
import argparse
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog

from data.data_loaders import get_val_augs, get_val_augs_multi_channel
# from models.unet import UNet, UNetResNet, UNetResNetAdapter
from models.unet import *
from config.enums import Strategy

# def get_models() -> dict[tuple[str, str, str], torch.nn.Module]:
#     return {
#         ("none", "none", "unet"): UNetA,
#         ("none", "resnet", "unet"): UNetA,
#         ("input", "resnet", "unet"): UNetA,
#         ("multi-layer", "resnet", "unet"): UNetA,
#         ("deep-input", "resnet", "unet"): UNetA,
#     }
def get_models() -> dict[tuple[str, str, str], torch.nn.Module]:
    return {
        ("none", "none", "unet"): UNet,
        ("none", "resnet", "unet"): UNetResNet,
        ("input", "resnet", "unet"): UNetResNet,
        ("multi-layer", "resnet", "unet"): UNetResNetAdapter,
        ("deep-input", "resnet", "unet"): UNetResNetAdapter,
    }

def get_augs() -> dict[str, Strategy]:
    return {
        "none": None,
        "aug": Strategy.BASIC,
        "multi": Strategy.MULTI,
    }

def strategy_type(x: str) -> Strategy:
    mapping = get_augs()
    if x not in mapping:
        raise argparse.ArgumentTypeError(f"Unknown strategy: {x}")
    return mapping[x]

def parse_arguments():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument(
        "-a", "--architecture",
        choices=["unet"],
        default="unet",
        help="Model Architecture"
    )
    # backbone
    parser.add_argument(
        "-b", "--backbone",
        choices=["none", "resnet"],
        default="none",
        help="Backbone Encoder"
    )
    # strategy
    # parser.add_argument(
    #     '-d', '--data-augmentation',
    #     choices=["aug", "multi"],
    #     default="aug",
    # )
    parser.add_argument(
        "-d", "--data-augmentation",
        type=strategy_type,
        default=Strategy.BASIC,
        help="Data Augmentation Policy:\n\taug  - 3 channel augmentation\n\tmulti - multi-channel augmentation"
    )
    # adapter
    parser.add_argument(
        '-m', '--mapper',
        choices=["none", "input", "deep-input", "multi-layer"],
        default="none",
        help="Adapter used to map inputs to the backbone:\n\tDefault: none: no adapter\n\tinput: a single-layer InputAdapter\n\tdeep-input: a multi-layer InputAdapter\n\tMultiLayer: a channel-grouped adapter."
    )
    # debug
    parser.add_argument(
        '--debug',
        choices=["normal", "verbose"],
        default=None,
    )
    args = parser.parse_args()

    if args.debug:
        print_arguments(args)

    return args

def print_arguments(args) -> None:
    config = vars(args)
    width = max(len(k) for k in config) + 5

    print('=' * 100)
    print(' ARGUMENTS '.center(100, '='))
    print('=' * 100)
    for key, value in config.items():
        print(f"{key + ':':.<{width}}{value}")
    print('=' * 100)

def open_image():
    root = tk.Tk()
    root.withdraw()

    filepath = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )

    image = Image.open(filepath).convert("RGB")

    return image

def get_image_tensor(aug):
    image = open_image()

    if aug == "aug":
        transform = get_val_augs()
    elif aug == "multi":
        transform = get_val_augs_multi_channel()

    if image:
        image = np.array(image)
        image = transform(image=image)["image"]
        image = image.unsqueeze(0)

    return image

def mIoU(preds, targets, num_classes, ignore_index=None):
    """
    Calculate mean IoU for semantic segmentation.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted logits/probabilities of shape (B, C, H, W)
        or class labels of shape (B, H, W).
    targets : torch.Tensor
        Ground-truth labels of shape (B, H, W).
    num_classes : int
        Number of segmentation classes.
    ignore_index : int, optional
        Label value to ignore.

    Returns
    -------
    float
        Mean IoU across classes present in the dataset.
    """
    
    # Convert logits to class predictions
    if preds.ndim == 4:
        preds = torch.argmax(preds, dim=1)

    preds = preds.view(-1)
    targets = targets.view(-1)

    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]

    ious = []

    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union == 0:
            # Skip classes absent from both prediction and target
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return float("nan")

    return torch.mean(torch.stack(ious)).item()
