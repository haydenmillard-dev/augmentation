import torch
import numpy as np
from PIL import Image

def pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Arguments:
        image: PIL.Image.Image with 3 channels
    Returns:
        torch.Tensor with shape [C, H, W]
    """
    image = torch.from_numpy(np.array(image))
    image = image.permute(2, 0, 1)
    return image.float() / 255.0

def pil_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """
    Arguments:
        mask: PIL.Image.Image with 1 channels
    Returns:
        torch.Tensor with shape [H, W]
    """
    return torch.from_numpy(np.array(mask)).long()

def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    """
    Arguments:
        image: torch.Tensor with shape [C, H, W] and values in [0, 1]
    Returns:
        PIL.Image.Image
    """
    image = image.clamp(0, 1)
    image = (image * 255).byte()

    # if grayscale
    if image.shape[0] == 1:
        image = image.squeeze(0)
        image = image.cpu().numpy()
        return Image.fromarray(image, mode="L")

    # image = image.permute(1, 2, 0)
    image = image.cpu().numpy()
    
    return Image.fromarray(image)

def tensor_to_PIL_image(image: torch.Tensor) -> Image.Image:
    """
    Arguments:
        image: torch.Tensor with shape [C, H, W] and values in [0, 1]
    Returns:
        PIL.Image.Image
    """
    image = image.detach().cpu().clamp(0, 1)

    # [C, H, W]
    if image.ndim == 3:
        c, h, w = image.shape

        # grayscale
        if c == 1:
            image = image.squeeze(0).numpy()
            return Image.fromarray((image * 255).astype(np.uint8), mode="L")

        # RGB
        image = image.permute(1, 2, 0).numpy()
        return Image.fromarray((image * 255).astype(np.uint8))

    raise ValueError(f"Expected input with [C ,H ,W] but received {image.shape}")

def tensor_to_pil_mask(mask: torch.Tensor) -> Image.Image:
    """
    Arguments:
        mask: torch.Tensor with shape [H, W]
    Returns:
        PIL.Image.Image with 1 channel
    """
    return Image.fromarray(mask.cpu().numpy().astype('uint8'))

def separate_representations(tensor: torch.Tensor, separable_channels: list[torch.Tensor]) -> torch.Tensor:
    """
    Arguments:
        tensor: the tensor containing the multiple representations in the shape [B,C,H,W] or [C,H,W].
        separable_channels: a list containing the channels that should be separated into individual tensors. The number of channels must equal the sum of the separable channels.
                            i.e., the number of tensor channels must equal sum(separable_channels).

    Returns:
        A list of tensors with the respective representations.

    Example:
        # Two representations of the same data
        >>> rep_1 = torch.tensor(8, 3, 256, 256)
        >>> rep_2 = torch.tensor(8, 3, 256, 256)
        >>> rep_3 = torch.tensor(8, 1, 256, 256)
        >>> concatenated_reps = torch.cat([rep1, rep_2, rep_3], dim=1)
        >>> concatenated_reps.shape
        [8, 6, 256, 256]
        # If two representations are contained in 3 channels respectively and one is contained in 1 channel:
        >>> sep_1, sep_2, sep_3 = separate_representations(concatenated_reps, [3, 3, 1])
        >>> sep_1.shape, sep_2.shape, sep_3.shape
        [8, 3, 256, 256], [8, 3, 256, 256], [8, 1, 256, 256]
        
    """
    shape_length = len(tensor.shape)
    # Adding batch dimension
    if shape_length == 3:
        tensor.unsqueeze(0)
    elif shape_length != 4:
        raise ValueError(
            f'Expected input shape of [B, C, H, W] but received shape {tensor.shape}.'
        )

    if sum(separable_channels) != tensor.shape[1]:
        raise ValueError(
            f'Input tensor with shape {tensor.shape} cannot be separated into {separable_channels} channels.'
        )
    
    start = 0
    end = 0
    separated_reps = []

    for c in separable_channels:
        end += c
        separated_reps.append(tensor[:,start:end])
        start = end

    return separated_reps

