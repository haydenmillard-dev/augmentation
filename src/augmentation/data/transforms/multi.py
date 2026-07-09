import torch
import kornia as K
import kornia.augmentation as KA
import kornia.filters as KF

import augmentation.data.transforms.augs as augs

_sequential_image = KA.AugmentationSequential(
    augs.verticalflip(),
    augs.horizontalflip(),
    keepdim=True,
    data_keys=["input"]
)

_sequential_image_mask = KA.AugmentationSequential(
    augs.verticalflip(),
    augs.horizontalflip(),
    keepdim=True,
    data_keys=["input", "mask"]
)

_normalise = augs.normalise()

_posterise = KA.RandomPosterize(bits=3, p=1.0)

_canny = KF.Canny(
#     low_threshold=0.25,
#     high_threshold=0.55,
#     kernel_size=(7, 7),
#     sigma=(1.5, 1.5),

    low_threshold=0.15,
    high_threshold=0.3,
    kernel_size=(5, 5),
    sigma=(0.8, 0.8),
)

def transform(image: torch.Tensor, mask: torch.Tensor=None, training_mode: bool=True):
    '''
    A transform that creates and stacks additional representations in the channel dimension.

    Required Arguments:
        image: torch.Tensor image of shape [B,C,H,W]
    Optional Arguments:
        mask: torch.Tensor mask of shape [B,H,W]

        training_mode: bool that determines whether positional transformations are applied.
    '''
    if training_mode:
        if mask is None:
            image = _sequential_image(image)
        else:
            image, mask = _sequential_image_mask(image, mask)

    posterised_image = _normalise(_posterise(image))
    normalised_image = _normalise(image)

    gray_image = K.color.rgb_to_grayscale(image)
    _, edges = _canny(gray_image)

    fused = torch.cat(
        [normalised_image, posterised_image, edges],
        dim=1,
    )

    if mask is None:
        return fused
    return fused, mask
