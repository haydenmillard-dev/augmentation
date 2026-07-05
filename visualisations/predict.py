import sys

from utils.utils import parse_arguments, get_image_tensor

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import kornia as K
import albumentations as A

from models.unet import UNetResNet, UNet, UNetResNetAdapter, UNetA
import visualisations.image_utils as iu
from config.enums import Strategy
import utils.utils as utils
import utils.tensor as ut
from aug.aug import augment

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(X, model):
    model.eval()

    with torch.inference_mode():
        logits = model(X)

    print(logits.shape)

    return torch.argmax(logits, dim=1)

def plot_prediction(prediction):
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.imshow(prediction.cpu().numpy())
    plt.show()

def plot_3_channel_prediction(input, prediction):
    pred = prediction.cpu().numpy()
    pred_rgb = iu.decode_segmap(pred, iu.voc_colormap())

    # Recover original image (denormalise)
    numpy_img = iu.to_numpy_img(input.squeeze(0)).clip(0, 1)
    denormed_img = iu.denormalise(input.squeeze(0)).clip(0, 1)
    denormed_img = iu.to_numpy_img(denormed_img)

    overlay_normed = iu.overlay(numpy_img, pred_rgb)
    overlay_denormed = iu.overlay(denormed_img, pred_rgb)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(denormed_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(overlay_normed)
    plt.title("Normalised")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_rgb)
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay_denormed)
    plt.title("Overlay")
    plt.axis("off")

    legend_patches = iu.create_legend(pred, iu.voc_colormap())
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

def plot_multi_channel_prediction(input, prediction):
    input = input.squeeze(0)

    rgb = input[0:3]
    posterised = input[3:6].permute(1, 2, 0).cpu().numpy()
    edges = input[6].cpu().numpy()

    pred = prediction.cpu().numpy()
    pred_rgb = iu.decode_segmap(pred, iu.voc_colormap())

    numpy_img = iu.to_numpy_img(rgb.squeeze(0)).clip(0, 1)
    denormed_img = iu.denormalise(rgb.squeeze(0)).clip(0, 1)
    denormed_img = iu.to_numpy_img(denormed_img)

    overlay_normed = iu.overlay(numpy_img, pred_rgb)
    overlay_denormed = iu.overlay(denormed_img, pred_rgb)

    edge_overlay = iu.overlay_edges_on_grayscale(denormed_img, edges, strength=0.5)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 4, 1)
    plt.imshow(denormed_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(numpy_img)
    plt.title("Normalised")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(posterised)
    plt.title("Posterised")
    plt.axis("off")
    
    plt.subplot(2, 4, 4)
    plt.imshow(edges)
    plt.title("Edges")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(edge_overlay)
    plt.title("Edge Overlay")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(overlay_denormed)
    plt.title("Overlay")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(overlay_normed)
    plt.title("Overlay Normalised")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.imshow(pred_rgb)
    plt.title("Mask")
    plt.axis("off")

    legend_patches = iu.create_legend(pred, iu.voc_colormap())
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

def prediction_augment(X, aug_strat):
    X = K.geometry.transform.Resize((256, 256))(X)
    return augment(X, strategy=aug_strat, training_mode=False)

if __name__ == '__main__':
    args = parse_arguments()

    # Model setup
    aug = args.data_augmentation
    num_classes = 21
    if aug is Strategy.BASIC:
        in_channels = 3
    elif aug is Strategy.MULTI:
        in_channels = 7
    else:
        ValueError(f"{aug} does not match {Strategy.BASIC} or {Strategy.MULTI}")
        print(f"Invalid agument: [-d] [--data-augmentation] {args.data_augmentation}")
    model = UNetA(adapter=adapter, backbone=backbone, in_channels=in_channels, num_classes=num_classes).to(DEVICE)
    
    append_model = f"{aug}_{adapter}_{backbone}_{architecture}"

    state_dict = torch.load(f"./checkpoints/best_{append_model}.pth", map_location=DEVICE)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(DEVICE)

    while True:
        image = utils.open_image()
        X = ut.pil_image_to_tensor(image).unsqueeze(0)

        if X is None:
            break
        X = prediction_augment(X, aug)

        prediction = predict(X.to(DEVICE), model).squeeze(0)
        
        for i in range(X.shape[0]):
            if aug is Strategy.BASIC:
                plot_3_channel_prediction(X, prediction)
            elif aug is Strategy.MULTI:
                plot_multi_channel_prediction(X, prediction)
            else:
                break
