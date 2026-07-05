import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

from visualisations.predict import plot_3_channel_prediction, plot_multi_channel_prediction, predict
import utils.utils as utils
from models.unet import UNet, UNetResNet, UNetResNetAdapter
from data.data_loaders import get_augmented_data_loaders, get_augmented_data_loaders_multi_channel
import visualisations.image_utils as iu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {
    ("unet", "none", "none"): UNet,
    ("unet", "resnet", "none"): UNetResNet,
    ("unet", "resnet", "input"): UNetResNet,
    ("unet", "resnet", "multi-layer"): UNetResNetAdapter,
    ("unet", "resnet", "deep-input"): UNetResNetAdapter,
}
BATCH_SIZE=8

args = utils.parse_arguments()

key = (args.architecture, args.backbone, args.mapper)
architecture, backbone, adapter = key
aug = args.data_augmentation

if key not in MODELS:
    keys = "\n".join(f"({arch}, {back}, {mapper})" for arch, back, mapper in MODELS.keys())
    parser.error(
        f"Invalid architecture-backbone combination:\n\t{key}\n"
        f"Expected a combination of the following:\n{keys}"
    )

# -----------------------
# Setup
# -----------------------
# train_loader, val_loader, test_loader = get_augmented_data_loaders()

# Model setup
aug = args.data_augmentation
if aug == "aug":
    train_loader, val_loader, test_loader = get_augmented_data_loaders(aug, batch_size=BATCH_SIZE)
    model = MODELS[key](num_classes=21, in_channels=3, adapter=adapter).to(DEVICE)
elif aug == "multi":
    train_loader, val_loader, test_loader = get_augmented_data_loaders_multi_channel(batch_size=BATCH_SIZE)
    model = MODELS[key](num_classes=21, in_channels=7, adapter=adapter).to(DEVICE)
else:
    print(f"Invalid agument: [-d] [--data-augmentation] {aug}")
    sys.exit(0)

# Load trained weights
checkpoint = torch.load(f"./checkpoints/best_model_{architecture}_{backbone}_{aug}_{adapter}.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print(model)

# -----------------------
# Visualisation
# -----------------------
cmap = iu.voc_colormap()
num_samples = 5

# loader = val_loader  # Use validation or test set instead
loader = test_loader  # Use validation or test set instead

def plot(img):
    # Denormalize for correct display
    img = iu.denormalise(X[j])
    img = iu.to_numpy_img(img)

    gt = y[j].cpu().numpy()
    pred = preds[j].cpu().numpy()

    gt_rgb = iu.decode_segmap(gt, cmap)
    pred_rgb = iu.decode_segmap(pred, cmap)

    overlay_pred = iu.overlay(img, pred_rgb)
    overlay_gt = iu.overlay(img, gt_rgb)

    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 5, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    # Ground truth
    plt.subplot(1, 5, 2)
    plt.imshow(gt_rgb)
    plt.title("GT Mask")
    plt.axis("off")

    # Prediction
    plt.subplot(1, 5, 3)
    plt.imshow(pred_rgb)
    plt.title("Prediction")
    plt.axis("off")

    # Overlay GT
    plt.subplot(1, 5, 4)
    plt.imshow(overlay_gt)
    plt.title("GT Overlay")
    plt.axis("off")

    # Overlay Prediction
    plt.subplot(1, 5, 5)
    plt.imshow(overlay_pred)
    plt.title("Pred Overlay")
    plt.axis("off")

    legend_patches = iu.create_legend(pred, cmap)
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

with torch.no_grad():
    for X, y in loader:

        # X.shape = [BATCH_SIZE, C, 256, 256]
        # y.shape = [BATCH_SIZE, 256, 256]
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        # prediction.shape = [BATCH_SIZE, 256, 256]
        prediction = predict(X, model).squeeze(0)

        # plot()
        for j in range(min(num_samples, X.size(0))):
            if aug == "aug":
                plot_3_channel_prediction(X[j], prediction[j])
            elif aug == "multi":
                plot_multi_channel_prediction(X[j].squeeze(0), prediction[j])

        should_i_continue = input("Continue?\n\tEnter *exit* to terminate the program. Enter anything else to continue.\n>")
        if should_i_continue == "exit":
            print("Exiting...")
            sys.exit(0)
