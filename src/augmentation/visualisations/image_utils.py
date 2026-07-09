import torch
import numpy as np
from matplotlib.patches import Patch

def voc_colormap(N=256):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = np.array([r, g, b])
    return cmap

def create_legend(mask, colormap=voc_colormap()):
    VOC_CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    patches = []

    unique_labels = np.unique(mask)

    for label in unique_labels:
        color = colormap[label] / 255.0

        patch = Patch(
            facecolor=color,
            edgecolor='black',
            label=VOC_CLASSES[label]
        )

        patches.append(patch)

    return patches

def decode_segmap(mask, colormap=voc_colormap(), ignore_label=255):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for label in np.unique(mask):
        if label == ignore_label:
            rgb[mask == label] = [0, 0, 0]
        else:
            rgb[mask == label] = colormap[label]

    return rgb

# -----------------------
# Helper: denormalise image (if needed)
# -----------------------
def show_image(img):
    img = img.permute(1, 2, 0).cpu().numpy()
    return img

def denormalise(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img.clamp(0, 1)

def to_numpy_img(img):
    return img.permute(1, 2, 0).cpu().numpy()

def overlay(image, mask_rgb, alpha=0.5):
    return (image * (1 - alpha) + mask_rgb / 255.0 * alpha)

def overlay_edges(image, edges, strength=0.7):
    image = image.copy()

    edges = edges.squeeze()
    edges = np.clip(edges, 0, 1)

    edges_rgb = np.stack([edges]*3, axis=-1)

    return np.clip(image + edges_rgb * strength, 0, 1)

def overlay_edges_on_grayscale(image, edges, strength=1.0):
    """
    Overlay edge map on a grayscale version of the image (red channel highlight style).

    Args:
        image (np.ndarray): RGB image in shape (H, W, 3), range [0, 1]
        edges (np.ndarray): Edge map in shape (H, W) or (1, H, W)
        strength (float): How strong the edge overlay should be

    Returns:
        np.ndarray: RGB image with edges highlighted
    """

    # Ensure correct shapes
    image = image.copy()
    edges = edges.squeeze()

    # Convert to grayscale
    gray = np.mean(image, axis=-1, keepdims=True)
    gray = np.repeat(gray, 3, axis=-1)

    # Ensure edges are in [0, 1]
    edges = np.clip(edges, 0, 1)

    # Create overlay
    overlay = gray.copy()

    # Highlight edges in red channel (paper-style visualization)
    overlay[..., 0] = np.clip(overlay[..., 0] + edges * strength, 0, 1)

    return overlay
