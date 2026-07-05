import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary

import utils.utils as utils
from data.loaders import get_data_loaders
from training.cost import CrossEntropyDiceLoss
from aug.aug import augment
from config.enums import Strategy
from config.train import (
    NUM_CLASSES,
    NUM_EPOCHS,
    BATCH_SIZE,
    MODELS,
    DEVICE,
)

def verify_model(key):
    if key not in MODELS:
        keys = "\n\t".join(
            f"({m}, {b}, {a})" for m, b, a in MODELS.keys()
        )
        print(
            f"Invalid architecture-backbone combination:\n\t{key}\n"
            f"Expected one of:\n\t{keys}"
        )
        sys.exit(1)
    return key

def config_unet(num_classes, key, aug):
    verify_model(key)

    adapter, backbone, arcitecture = key
    if aug == Strategy.BASIC:
        model = MODELS[key](adapter=adapter, backbone=backbone, in_channels=3, num_classes=num_classes).to(DEVICE)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    elif aug == Strategy.MULTI:
        model = MODELS[key](adapter=adapter, backbone=backbone, in_channels=7, num_classes=num_classes).to(DEVICE)

        # adapter_params = list(model.adapter.parameters())
        # encoder_params = (
        #     list(model.encoder0.parameters()) +
        #     list(model.encoder1.parameters()) +
        #     list(model.encoder2.parameters()) +
        #     list(model.encoder3.parameters()) +
        #     list(model.encoder4.parameters())
        # )
        # decoder_params = (
        #     list(model.decoder1.parameters()) +
        #     list(model.decoder2.parameters()) +
        #     list(model.decoder3.parameters()) +
        #     list(model.decoder4.parameters()) +
        #     list(model.final.parameters())
        # )

        # optimizer = torch.optim.AdamW(
        #     [
        #         {"params": adapter_params, "lr": 1e-3},
        #         {"params": encoder_params, "lr": 1e-5},
        #         {"params": decoder_params, "lr": 1e-4},
        #     ],
        #     weight_decay=1e-4,
        # )

    else:
        print(f"Invalid argument: [-d] [--data-augmentation] {aug}")
        sys.exit(1)

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
        X, y = augment(X, y, aug)

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
            X, y = augment(X, y, aug)

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

#     images, masks = next(iter(train_loader))
#     summary(model, input_data=(images, masks))

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
            f"./checkpoints/best_{model_name}.pth"
            if is_best
            else f"./checkpoints/last_{model_name}.pth"
        )
        save_checkpoint(
            checkpoint_path, model, optimizer, epoch,
            model_name, train_losses, val_losses, epoch_list, best_val_loss,
        )


if __name__ == "__main__":
    main()
