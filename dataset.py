import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BrainTumorDataset(Dataset):
    """
    loads brain MRI images and their tumor masks.

    Each MRI has a matching mask file:
      - MRI image: (the brain scan)
      - Mask: (white = tumor, black = no tumor)
    """

    def __init__(self, data_dir, image_size=256):
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_paths = []  # stores paths to MRI images
        self.mask_paths = []  # stores paths to matching masks

        # Walk through all patient folders and find image/mask pairs
        for patient_folder in sorted(os.listdir(data_dir)):
            patient_path = os.path.join(data_dir, patient_folder)

            if not os.path.isdir(patient_path):
                continue

            # Get all files in this patient's folder
            files = sorted(os.listdir(patient_path))

            for file in files:
                # Skip mask files here — we'll load them separately
                if "_mask" in file:
                    continue
                #Skipping mask files, but checking if the image name contains a matching mask
                # Check if this image has a matching mask
                name, ext = os.path.splitext(file)
                mask_file = name + "_mask" + ext
                #if an image and mask pair is found, append them to their separate lists (image/mask_paths
                if mask_file in files:
                    self.image_paths.append(os.path.join(patient_path, file))
                    self.mask_paths.append(os.path.join(patient_path, mask_file))

        print(f"Found {len(self.image_paths)} image-mask pairs")

    def __len__(self):
        return len(self.image_paths)
    #length of the image paths

    def __getitem__(self, idx):
        #this is the get image and mask pair where it tells the code how to do this once
        # Load the MRI image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Load the mask (grayscale — just black and white)
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Resize both to same size
        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size))
        #Now that you have resized and loaded both the images after importing them,
        # you must convert them into numpy arrays for the model to scan them
        # Convert to numpy arrays
        image = np.array(image, dtype=np.float32) / 255.0  # normalize to 0-1
        mask = np.array(mask, dtype=np.float32) / 255.0  # normalize to 0-1
        #normalize both them to 0-1 for the mask to filter out low priority pixels (under 0.5 brightness)
        # Make mask binary: 1 = tumor, 0 = no tumor
        mask = (mask > 0.5).astype(np.float32)

        # Rearrange image from (H, W, 3) to (3, H, W) — PyTorch expects this
        image = np.transpose(image, (2, 0, 1))

        # Add channel dimension to mask: (H, W) → (1, H, W)
        mask = np.expand_dims(mask, axis=0)

        return image, mask


# Quick test — run this file directly to see if data loads
if __name__ == "__main__":
    dataset = BrainTumorDataset("brainTumorProj/kaggle_3m")
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")  # should be (3, 256, 256)
    print(f"Mask shape: {mask.shape}")  # should be (1, 256, 256)
    print(f"Image range: {image.min():.2f} to {image.max():.2f}")
    print(f"Mask unique values: {np.unique(mask)}")