import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary

from augmentation.utils import utils
from augmentation.data.loaders import get_data_loaders
from augmentation.training import CrossEntropyDiceLoss
from augmentation.models.unet import UNet
from augmentation.data.augment import apply_transform
from augmentation.config.enums import Strategy
from augmentation.config.paths import CHECKPOINTS
from augmentation.config.train import (
    NUM_CLASSES,
    NUM_EPOCHS,
    BATCH_SIZE,
    DEVICE,
)

def config_unet(num_classes, key, aug):
    adapter, backbone, arcitecture = key
    if aug == Strategy.BASIC:
        in_channels = 3
    elif aug == Strategy.MULTI:
        in_channels = 7
    else:
        print(f"Invalid argument: [-d] [--data-augmentation] {aug}")
        sys.exit(1)

    model = UNet(adapter=adapter, backbone=backbone, in_channels=in_channels, num_classes=num_classes).to(DEVICE)

    if adapter is None or backbone == "none":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    elif backbone == "resnet":
        adapter_params, encoder_params, decoder_params, final_params = model.get_module_params()
        optimizer = torch.optim.AdamW(
            [
                {"params": adapter_params, "lr": 1e-3},
                {"params": encoder_params, "lr": 1e-5},
                {"params": decoder_params, "lr": 1e-4},
                {"params": final_params, "lr": 1e-4},
            ],
            weight_decay=1e-4,
        )
    else:
        print(f"Invalid argument: [-b] [--backbone] {backbone}")
        sys.exit(1)

    return model, optimizer

def train_epoch(model, optimizer, loss_fn, train_loader, aug):
    """Runs one training epoch, returns average loss."""
    model.train()
    total_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        X, y = apply_transform(X, y, aug)

        loss = loss_fn(model(X), y.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.shape[0]

    return total_loss / len(train_loader.dataset)

def val_epoch(model, loss_fn, val_loader, aug):
    """Runs one validation epoch, returns average loss."""
    model.eval()
    total_loss = 0.0

    with torch.inference_mode():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            X, y = apply_transform(X, y, aug, training_mode=False)

            loss = loss_fn(model(X), y.long())
            total_loss += loss.item() * X.shape[0]

    return total_loss / len(val_loader.dataset)

def plot_graph(ax, model_name, epoch_list, train_losses, val_losses):
    """Saves an updated loss curve to disk."""
    ax.clear()
    ax.plot(epoch_list, train_losses, label="Train Loss")
    ax.plot(epoch_list, val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.set_title("Training Progress")
    ax.legend()
    os.makedirs("./graphs", exist_ok=True)
    plt.savefig(f"./graphs/{model_name}.png")

def save_checkpoint(path, model, optimizer, epoch, model_name, train_losses, val_losses, epoch_list, val_loss):
    """Saves model + optimizer state and training history to a checkpoint."""
    architecture, backbone, adapter = (
        model_name.split("_")[1],
        model_name.split("_")[2],
        model_name.split("_")[0],
    )
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "backbone": backbone,
            "adapter": adapter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch_list": epoch_list,
        },
        path,
    )

def main():
    args = utils.parse_arguments()
    adapter = args.mapper
    backbone = args.backbone
    architecture = args.architecture
    aug = args.data_augmentation
    key = (adapter, backbone, architecture)

    if architecture == "unet":
        model, optimizer = config_unet(NUM_CLASSES, key, aug)
    else:
        print(f'Invalid argument: [-a] [--architecture] given [{architecture}]')
        sys.exit(1)

    train_loader, val_loader, _ = get_data_loaders(batch_size=BATCH_SIZE)
    loss_fn = CrossEntropyDiceLoss(ce_split=0.7)

    train_losses, val_losses, epoch_list = [], [], []
    best_val_loss = float("inf")
    
    if aug == Strategy.BASIC:
        aug_strategy = "basic"
    if aug == Strategy.MULTI:
        aug_strategy = "multi"
    else:
        ValueError(f"{aug} does not match {Strategy.BASIC} or {Strategy.MULTI}")
    model_name = f"{aug_strategy}_{adapter}_{backbone}_{architecture}"

    _, ax = plt.subplots()

    for epoch in range(NUM_EPOCHS):
        avg_train_loss = train_epoch(model, optimizer, loss_fn, train_loader, aug)
        avg_val_loss = val_epoch(model, loss_fn, val_loader, aug)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epoch_list.append(epoch)

        print(
            f"Epoch: {epoch:>3} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        plot_graph(ax, model_name, epoch_list, train_losses, val_losses)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        checkpoint_path = (
            CHECKPOINTS / f"best_{model_name}.pth"
            if is_best
            else CHECKPOINTS / f"last_{model_name}.pth"
        )
        save_checkpoint(
            checkpoint_path, model, optimizer, epoch,
            model_name, train_losses, val_losses, epoch_list, best_val_loss,
        )


if __name__ == "__main__":
    main()
