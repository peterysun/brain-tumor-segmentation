import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import BrainTumorDataset
from model import UNet


def dice_score(pred, target):
    """
    Dice Score: measures overlap between prediction and ground truth.
    1.0 = perfect match
    0.0 = no overlap at all
    """
    smooth = 1e-5  # avoid division by zero
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def train():
    DATA_DIR = "brainTumorProj/kaggle_3m"
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = BrainTumorDataset(DATA_DIR, image_size=IMAGE_SIZE)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


    model = UNet().to(device)


    criterion = nn.BCELoss()


    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    val_dice_scores = []

    print("\n--- Starting Training ---\n")

    for epoch in range(EPOCHS):

        model.train()
        epoch_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):

            images = images.float().to(device)
            masks = masks.float().to(device)

            predictions = model(images)

            loss = criterion(predictions, masks)

            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        val_dice = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.float().to(device)
                masks = masks.float().to(device)

                predictions = model(images)
                loss = criterion(predictions, masks)
                val_loss += loss.item()

                pred_binary = (predictions > 0.5).float()
                val_dice += dice_score(pred_binary, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_val_dice)

        print(f"  Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Dice: {avg_val_dice:.4f}")

    torch.save(model.state_dict(), "brain_tumor_model.pth")
    print("\nModel saved to brain_tumor_model.pth")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(val_dice_scores, label="Val Dice Score", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()
    print("Training curves saved to training_curves.png")


if __name__ == "__main__":
    train()
