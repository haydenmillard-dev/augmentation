import torch
import kornia.augmentation as KA

def resize():
    return KA.Resize((256, 256))

def verticalflip():
    return KA.RandomVerticalFlip()

def horizontalflip():
    return KA.RandomHorizontalFlip()

def normalise():
    return KA.Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]))
