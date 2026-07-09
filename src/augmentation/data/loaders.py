import torch
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, Dataset, random_split

from augmentation.data.dataset import SegmentationDataset
from augmentation.config.loaders import (
    DATA_DIR,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
)
from augmentation.config.paths import DATASETS

def load_dataset():
    return VOCSegmentation(
        root=DATASETS,
        year="2012",
        image_set="trainval",
        download=True,
    )

def create_splits(
        dataset: Dataset,
        seed: int = 42,
    ) -> list[Dataset, Dataset, Dataset]:
    return random_split(
        dataset,
        [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT],
        generator=torch.Generator().manual_seed(seed)
    )

def get_loader(dataset, batch_size, shuffle, size=(256, 256), seed=None):
    dataset = SegmentationDataset(dataset, size=size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        generator=seed,
    )

def get_train_loader(train_set, batch_size, seed=None):
    return get_loader(train_set, batch_size, shuffle=True, seed=seed)

def get_eval_loader(eval_set, batch_size, seed=None):
    return get_loader(eval_set, batch_size, shuffle=False, seed=seed)

def get_data_loaders(
        batch_size: int
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = load_dataset()
    train_set, val_set, test_set = create_splits(dataset)

    train_loader = get_train_loader(train_set, batch_size)
    val_loader = get_eval_loader(val_set, batch_size)
    test_loader = get_eval_loader(test_set, batch_size)

    return train_loader, val_loader, test_loader
