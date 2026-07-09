import torch
import kornia as K
import kornia.augmentation as KA

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

def transform(input, mask=None, training_mode: bool=True):
    input = _normalise(input)

    if training_mode:
        if mask is None:
            return _sequential_image(input)
        else:
            return _sequential_image_mask(input, mask)

    if mask is None:
        return input
    return input, mask
