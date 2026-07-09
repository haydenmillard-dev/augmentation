import torch
import matplotlib.pyplot as plt
from data.data_loaders import get_augmented_data_loaders
from data.loaders import get_data_loaders
import visualisations.image_utils as iu
from config.enums import Strategy
from aug.aug import augment

# _, val_loader, _ = get_augmented_data_loaders()
batch_size = 8
train_loader, val_loader, test_loader = get_data_loaders(batch_size)
image_batch, mask_batch = next(iter(val_loader))

# batch_size = image_batch.shape[0]
num_samples = 4
iterations = batch_size // num_samples

for k in range(iterations):
    plt.figure(figsize=(12, 12))

    for i in range(num_samples):
        # Denormalised Image
        image = image_batch[k * num_samples + i]
        # training_mode=True to simulate what the augmentation might look like during training
        image = augment(image, training_mode=True)
        denormed_image = iu.denormalise(image).permute(1, 2, 0)
        plt.subplot(num_samples, 4, 4 * i + 1)
        plt.title(f"Image {k * num_samples + i}")
        plt.imshow(denormed_image)
        plt.axis("off")

        # Normalised Image
        normed_image = image.permute(1, 2, 0).clamp(0, 1)
        plt.subplot(num_samples, 4, 4 * i + 2)
        plt.title(f"Normalised Image {k * num_samples + i}")
        plt.imshow(normed_image)
        plt.axis("off")

        # Mask
        mask = mask_batch[k * num_samples + i]
        mask[mask == 255] = 0
        mask = iu.decode_segmap(mask, iu.voc_colormap())
        plt.subplot(num_samples, 4, 4 * i + 3)
        plt.title(f"Mask {k * num_samples + i}")
        plt.imshow(mask)
        plt.axis("off")

        # Mask over Image
        plt.subplot(num_samples, 4, 4 * i + 4)
        plt.title(f"Masked Image {k * num_samples + i}")
        plt.imshow(denormed_image)
        plt.imshow(mask, alpha=0.4)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
