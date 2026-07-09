import numpy as np
import torch
from torch.utils.data import Dataset
import kornia as K

import augmentation.utils.tensor as ut

class SegmentationDataset(Dataset):
    def __init__(self, dataset, size=(256, 256)):
        self.dataset = dataset

        self.img_resize = K.geometry.transform.Resize(size)
        self.mask_resize = K.geometry.transform.Resize(size, interpolation="nearest")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        image = K.image_to_tensor(np.array(image)).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).long()

        image = self.img_resize(image.unsqueeze(0)).squeeze(0)

        mask = self.mask_resize(mask.unsqueeze(0).unsqueeze(0).float())
        mask = mask.squeeze(0).squeeze(0).long()

        return image, mask
