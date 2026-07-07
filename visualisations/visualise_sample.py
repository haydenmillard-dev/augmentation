import torch
import numpy as np
import matplotlib.pyplot as plt

from data.loaders import load_dataset
from config.enums import Strategy
import utils.tensor as ut
from utils.parse import visualise_sample_args
from data.transforms.multi import transform as multi_transform
from pathlib import Path
from PIL import Image

def basic(dataset, index):
    pass

def plot_image(image: Image.Image, augment: callable):
    image_tensor = ut.pil_image_to_tensor(image)
    augmented = augment(image_tensor.unsqueeze(0))

    normalised_image, posterised_image, edges = ut.separate_representations(augmented, [3, 3, 1])

    normalised_image = ut.tensor_to_PIL_image(normalised_image.squeeze())
    posterised_image = ut.tensor_to_PIL_image(posterised_image.squeeze())
    edges = ut.tensor_to_pil_mask(edges.squeeze())

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(normalised_image)
    plt.title("Normalised")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(posterised_image)
    plt.title("Posterised")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(edges)
    plt.title("Edges")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def multi(example):
    while True:
        if isinstance(example, int):
            dataset = load_dataset()
            image, mask = dataset[example]
            plot_image(image, multi_transform)
        elif isinstance(example, Path):
            image = Image.open(example)
            plot_image(image, multi_transform)

        response = input("Terminate program?\n")
        if response == 'y':
            return
        else:
            return

def main():
    args = visualise_sample_args()
    aug = args.strategy
    
    if aug == Strategy.BASIC:
        basic(args.example)
    elif aug == Strategy.MULTI:
        multi(args.example)

if __name__=='__main__':
    '''
    python -m visualisations.visualise_sample -e 7 -s multi
    python -m visualisations.visualise_sample -e path/to/file -s basic
    
    if -e --example is a number it will select the index from the dataset
    if -e --example is a file path it will open the given file
    '''
    main()
