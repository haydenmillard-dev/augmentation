import torch

from config.enums import Strategy
from data.transforms.basic import transform as basic_transform
from data.transforms.multi import transform as multi_transform
# from data.transforms.advanced import transform as advanced_transform

TRANSFORMS = {
    Strategy.BASIC: basic_transform,
    Strategy.MULTI: multi_transform,
    # Strategy.ADVANCED: advanced_transform,
}

def get_transform(strategy: Strategy):
    """
    Returns a callable transform for the specific strategy.
    Raises a KeyError if the strategy does not exist.
    """
    try:
        return TRANSFORMS[strategy]
    except KeyError:
        raise ValueError(f"Unknown strategy: {strategy}")

def augment(image: torch.Tensor, mask: torch.Tensor = None, strategy: Strategy = Strategy.BASIC, training_mode=True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the augmented image (B,C,H,W) and mask (B, H, W) torch.Tensors

    Required Argument:
        image: torch.Tensor of shape [B,C,H,W]

    Optional Arguments:
        mask: torch.Tensor of shape [B,H,W]

        strategy: Strategy that determines the type of augmentation to be applied

        training_mode: bool that turns on positional transforms when true and turns them off when False 
    """
    transform = get_transform(strategy)
    if mask is None:
        return transform(image, training_mode=training_mode)
    return transform(image, mask, training_mode=training_mode)
