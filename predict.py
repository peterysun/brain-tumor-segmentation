import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, random_split

from dataset import BrainTumorDataset
from model import UNet


def visualize_predictions(num_samples=8):

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=device))
    model.eval()
    print("Model loaded!")

    # Load data
    dataset = BrainTumorDataset("brainTumorProj/kaggle_3m", image_size=256)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Find samples that actually have tumors
    tumor_indices = []
    no_tumor_indices = []

    for i in range(len(val_dataset)):
        _, mask = val_dataset[i]
        if mask.sum() > 0:
            tumor_indices.append(i)
        else:
            no_tumor_indices.append(i)

    print(f"Validation samples with tumors: {len(tumor_indices)}")
    print(f"Validation samples without tumors: {len(no_tumor_indices)}")

    # mix
    show_indices = tumor_indices[:num_samples]


    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(show_indices):
        image, mask = val_dataset[idx]


        image_tensor = torch.tensor(image).unsqueeze(0).float().to(device)
        with torch.no_grad():
            prediction = model(image_tensor)


        image_np = np.transpose(image, (1, 2, 0))  # (3,H,W) → (H,W,3)
        mask_np = mask.squeeze()  # (1,H,W) → (H,W)
        pred_np = prediction.cpu().squeeze().numpy()  # (1,1,H,W) → (H,W)

        # original MRI
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("MRI Scan", fontsize=12)
        axes[i, 0].axis("off")

        # ground truth mask
        axes[i, 1].imshow(mask_np, cmap="Reds")
        axes[i, 1].set_title("Actual Tumor", fontsize=12)
        axes[i, 1].axis("off")

        # model prediction
        axes[i, 2].imshow(pred_np, cmap="Reds")
        axes[i, 2].set_title("AI Prediction", fontsize=12)
        axes[i, 2].axis("off")

    plt.suptitle("Brain Tumor Segmentation Results", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    plt.show()
    print("Predictions saved to predictions.png")


def overlay_prediction(image_path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=device))
    model.eval()

    # Load a sample
    dataset = BrainTumorDataset("brainTumorProj/kaggle_3m", image_size=256)

    # Find one with a tumor
    for i in range(len(dataset)):
        image, mask = dataset[i]
        if mask.sum() > 0:
            break


    image_tensor = torch.tensor(image).unsqueeze(0).float().to(device)
    with torch.no_grad():
        prediction = model(image_tensor)


    image_np = np.transpose(image, (1, 2, 0))
    pred_np = prediction.cpu().squeeze().numpy()
    pred_binary = (pred_np > 0.5).astype(np.float32)

    # Create overlay
    overlay = image_np.copy()
    overlay[pred_binary == 1, 0] = 1.0  # red channel
    overlay[pred_binary == 1, 1] *= 0.3  # dim green
    overlay[pred_binary == 1, 2] *= 0.3  # dim blue

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image_np)
    ax1.set_title("Original MRI")
    ax1.axis("off")

    ax2.imshow(overlay)
    ax2.set_title("AI Tumor Detection (Red = Tumor)")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig("overlay.png", dpi=150)
    plt.show()
    print("Overlay saved to overlay.png")


if __name__ == "__main__":
    print("=== Generating Predictions ===\n")
    visualize_predictions(num_samples=6)

    print("\n=== Generating Overlay ===\n")
    overlay_prediction()
