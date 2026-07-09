import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

from rich import print

from augmentation.visualisations.predict import plot_3_channel_prediction, plot_multi_channel_prediction, predict
import augmentation.utils.utils as utils
from augmentation.models.unet import UNet
from augmentation.data.loaders import get_data_loaders
import augmentation.visualisations.image_utils as iu
from augmentation.config.enums import Strategy
from augmentation.config.paths import CHECKPOINTS
from augmentation.data.augment import apply_transform
from augmentation.data.transforms.augs import resize

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

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE=8

    args = utils.parse_arguments()

    architecture, backbone, adapter = args.architecture, args.backbone, args.mapper
    aug = args.data_augmentation

    if aug is Strategy.BASIC:
        in_channels = 3
        aug_name = 'basic'
    elif aug is Strategy.MULTI:
        in_channels = 7
        aug_name = 'multi'
    else:
        print(f"Invalid agument: [-d] [--data-augmentation] {aug}")
        sys.exit(0)

    # Load trained weights
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)
    path = CHECKPOINTS / f'best_{aug_name}_{adapter}_{backbone}_{architecture}.pth'
    # checkpoint = torch.load(CHECKPOINTS / f"best_model_{architecture}_{backbone}_{aug_name}_{adapter}.pth", map_location=DEVICE)
    # print(f'[red].\\augmentation\\checkpoints\\best_{aug_name}_{adapter}_{backbone}_{architecture}.pth[/red]')
    print(f'[red]{path}[/red]')
    # checkpoint = torch.load(f'./augmentation/checkpoints/best_{aug_name}_{adapter}_{backbone}_{architecture}.pth', map_location=DEVICE)
    # path = 'D:\\Honours\\research\\src\\checkpoints\\best_multi_deep-input_resnet_unet.pth'
    checkpoint = torch.load(path, map_location=DEVICE)

    model = UNet(adapter=adapter, backbone=backbone, in_channels=in_channels, num_classes=21).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # -----------------------
    # Visualisation
    # -----------------------
    cmap = iu.voc_colormap()
    num_samples = 5

    loader = test_loader


    with torch.no_grad():
        for X, y in loader:

            # X.shape = [BATCH_SIZE, C, 256, 256]
            # y.shape = [BATCH_SIZE, 256, 256]
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            X = apply_transform(resize()(X), strategy=aug, training_mode=False)
            # prediction.shape = [BATCH_SIZE, 256, 256]
            prediction = predict(X, model).squeeze(0)

            # plot()
            for j in range(min(num_samples, X.size(0))):
                if aug is Strategy.BASIC:
                    plot_3_channel_prediction(X[j], prediction[j])
                elif aug is Strategy.MULTI:
                    plot_multi_channel_prediction(X[j].squeeze(0), prediction[j])

            should_i_continue = input("Continue?\n\tEnter *exit* to terminate the program. Enter anything else to continue.\n>")
            if should_i_continue == "exit":
                print("Exiting...")
                sys.exit(0)

if __name__ == '__main__':
    main()
