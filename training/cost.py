import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.shape[1]

        valid_mask = (targets != self.ignore_index)

        targets = targets.clone()
        targets[~valid_mask] = 0

        targets_one_hot = F.one_hot(targets, num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        probs = F.softmax(logits, dim=1)

        valid_mask = valid_mask.unsqueeze(1)

        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, ce_split, ignore_index=255):
        super().__init__()
        self.ce_split = ce_split
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=255)

    def forward(self, logits, target):
        return self.ce_split * self.ce(logits, target) + (1 - self.ce_split) * self.dice(logits, target)
