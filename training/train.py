import argparse
import os
import sys

# from data.data_loaders import get_data_loaders
from data.data_loaders import get_augmented_data_loaders, get_augmented_data_loaders_multi_channel
from models.unet import UNet, UNetResNet, UNetResNetAdapter
from training.cost import CrossEntropyDiceLoss

import torch
import torch.nn as nn
from torch.optim import AdamW

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import utils.utils as utils

def print_config(architecture, backbone, adapter, data_augmentation):
    print(architecture)
    print(backbone)
    print(adapter)
    print(data_augmentation)

# Constants
NUM_EPOCHS = 100
BATCH_SIZE=16
MODELS = {
    ("unet", "none", "none"): UNet,
    ("unet", "resnet", "none"): UNetResNet,
    ("unet", "resnet", "input"): UNetResNet,
    ("unet", "resnet", "multi-layer"): UNetResNetAdapter,
    ("unet", "resnet", "deep-input"): UNetResNetAdapter,
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = utils.parse_arguments()

key = (args.architecture, args.backbone, args.mapper)
architecture, backbone, adapter = key
if key not in MODELS:
    keys = "\n".join(f"({arch}, {back}, {mapper})" for arch, back, mapper in MODELS.keys())
    parser.error(
        f"Invalid architecture-backbone combination:\n\t{key}\n"
        f"Expected a combination of the following:\n{keys}"
    )

# Model setup
aug = args.data_augmentation
if aug == "aug":
    train_loader, val_loader, test_loader = get_augmented_data_loaders(batch_size=BATCH_SIZE)
    if adapter == "none":
        model = MODELS[key](num_classes=21, in_channels=3).to(DEVICE)
    else:
        model = MODELS[key](num_classes=21, in_channels=3, adapter=adapter).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
elif aug == "multi":
    train_loader, val_loader, test_loader = get_augmented_data_loaders_multi_channel(batch_size=BATCH_SIZE)
    model = MODELS[key](num_classes=21, in_channels=7, adapter=adapter).to(DEVICE)
    
    adapter_params = (list(model.adapter.parameters()))

    encoder_params = (
        list(model.encoder0.parameters()) +
        list(model.encoder1.parameters()) +
        list(model.encoder2.parameters()) +
        list(model.encoder3.parameters()) +
        list(model.encoder4.parameters())
    )

    decoder_params = (
        list(model.decoder1.parameters()) +
        list(model.decoder2.parameters()) +
        list(model.decoder3.parameters()) +
        list(model.decoder4.parameters()) +
        list(model.final.parameters())
    )
    
    optimizer = AdamW(
        [
            {"params": adapter_params, "lr": 1e-3},
            {"params": encoder_params, "lr": 1e-5},
            {"params": decoder_params, "lr": 1e-4},
        ],
        weight_decay=1e-4
    )
else:
    print(f"Invalid agument: [-d] [--data-augmentation] {aug}")
    sys.exit(0)

#print_config(architecture, backbone, adapter, aug)
loss_fn = CrossEntropyDiceLoss(ce_split=0.7)

start_epoch = 0
train_losses = []
val_losses = []
epoch_list = []
best_val_loss = float("inf")

fig, ax = plt.subplots()

# Training-Validation Loop
for epoch in range(start_epoch, NUM_EPOCHS):
    # Training
    model.train()
    train_loss_total = 0

    for X, y in train_loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(X)
        loss = loss_fn(logits, y.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * X.shape[0]

    avg_train_loss = train_loss_total / len(train_loader.dataset)

    # Testing
    model.eval()
    val_loss_total = 0

    with torch.inference_mode():
        for X, y in val_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
            validation_loss = loss_fn(logits, y.long())
            val_loss_total += validation_loss.item() * X.shape[0]

    avg_val_loss = val_loss_total / len(val_loader.dataset)

    print(f"Epoch: {epoch:>3} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

    # Update training and validation plot
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    epoch_list.append(epoch)

    ax.clear()
    ax.plot(epoch_list, train_losses, label="Train Loss")
    ax.plot(epoch_list, val_losses, label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.set_title("Training Progress")
    ax.legend()

    append_model = f"{architecture}_{backbone}_{aug}_{adapter}"
    plt.savefig(f"./graphs/{append_model}.png")

    # Saving the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        torch.save({
            "architecture": architecture,
            "backbone": backbone,
            "adapter": adapter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch_list": epoch_list,
        }, f"./checkpoints/best_model_{append_model}.pth")
    # Or saving the last epoch in case training terminates
    else:
        torch.save({
            "architecture": architecture,
            "backbone": backbone,
            "adapter": adapter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch_list": epoch_list,
        }, f"./checkpoints/last_{append_model}.pth")
