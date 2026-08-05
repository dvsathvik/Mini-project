import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_markdown(text):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split('\n')]
        })

    def add_code(code):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.split('\n')]
        })

    add_markdown("# Medical Image Segmentation (Hippocampus)\nThis notebook implements a fully 3D U-Net using MONAI to precisely segment the hippocampus from MRI volumes, acting as a biomarker measurement tool for early Alzheimer's disease.")
    
    add_code("""# Install required specialized medical AI libraries
!pip install -q monai nibabel plotly""")

    add_markdown("## 1. Imports and Setup")
    add_code("""import os
import glob
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityRangePercentilesd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, ToTensord, CropForegroundd,
    SpatialPadd
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

set_determinism(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")""")

    add_markdown("## 2. Dataset Preparation (Auto-Download)\nWe use MONAI's built-in `DecathlonDataset` API to automatically download the Medical Segmentation Decathlon Hippocampus dataset directly into the Kaggle environment. **You don't need to manually search for or attach any datasets!**")
    add_code("""import monai
# Define Kaggle working directory for downloading
data_dir = '/kaggle/working/'

print("Downloading MSD Hippocampus Dataset (This takes ~30 seconds)...")
# This triggers the automatic download and extraction from the official medical servers
dummy_ds = monai.apps.DecathlonDataset(
    root_dir=data_dir,
    task="Task04_Hippocampus",
    section="training",
    download=True,
)

extracted_dir = os.path.join(data_dir, "Task04_Hippocampus")
train_images = sorted(glob.glob(os.path.join(extracted_dir, 'imagesTr', '*.nii.gz')))
train_labels = sorted(glob.glob(os.path.join(extracted_dir, 'labelsTr', '*.nii.gz')))

# Split 80/20 for train/validation
val_split = int(len(train_images) * 0.2)
val_images = train_images[:val_split]
val_labels = train_labels[:val_split]
train_images = train_images[val_split:]
train_labels = train_labels[val_split:]

train_files = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]
val_files = [{"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)]

print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")""")

    add_markdown("## 3. MONAI 3D Data Pipeline (Transforms)\nMedical 3D volumes are too large for standard GPUs. We crop them and extract randomized 3D patches.")
    add_code("""# Define transforms for Training
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    # Crop to just the foreground (removes empty black space)
    CropForegroundd(keys=["image", "label"], source_key="image"),
    # Pad images that are smaller than our patch size (32,32,32) to prevent cropping errors
    SpatialPadd(keys=["image", "label"], spatial_size=(32, 32, 32)),
    # Normalize intensities to improve model stability
    ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True),
    # Extract random 3D patches from the volume (ensures balanced background/foreground)
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(32, 32, 32), # Patch size to fit in VRAM
        pos=1,
        neg=1,
        num_samples=4, # Extract 4 patches per volume per batch
    ),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
    ToTensord(keys=["image", "label"]),
])

# Define transforms for Validation (No random cropping/flipping)
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=(32, 32, 32)),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image", "label"]),
])

# Create MONAI Datasets and DataLoaders
# CacheDataset preloads data to RAM for massive speedups
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)""")

    add_markdown("## 4. Initialize 3D U-Net Model and Loss")
    add_code("""# Standard Medical 3D UNet architecture
model = UNet(
    spatial_dims=3, # 3D Medical Imaging
    in_channels=1,
    out_channels=3, # 0: Background, 1: Hippocampus Head, 2: Hippocampus Body
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# Dice Loss is the standard for Segmentation (optimizes Intersection over Union)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
dice_metric = DiceMetric(include_background=False, reduction="mean")""")

    add_markdown("## 5. Training Loop\nWe train for a limited number of epochs to demonstrate convergence. On Kaggle, you can run this for 100+ epochs.")
    add_code("""max_epochs = 20
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
epoch_loss_std_values = []
metric_values = []
metric_std_values = []

print("Starting Training...")
for epoch in range(max_epochs):
    model.train()
    step = 0
    batch_losses = []
    
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        
    epoch_loss = np.mean(batch_losses)
    epoch_std = np.std(batch_losses)
    epoch_loss_values.append(epoch_loss)
    epoch_loss_std_values.append(epoch_std)
    print(f"Epoch {epoch + 1}/{max_epochs} - Loss: {epoch_loss:.4f} ± {epoch_std:.4f}")
    
    # Validation Phase
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                
                # Because validation volumes are full size, we must use sliding window inference
                val_outputs = sliding_window_inference(
                    val_inputs, (32, 32, 32), 4, model
                )
                
                # Convert outputs to discrete one-hot format for Dice metric
                val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                dice_metric(y_pred=val_outputs, y=val_labels)
                
            # Access the raw buffer to calculate mean AND standard deviation across all volumes
            scores = dice_metric.get_buffer()
            if torch.is_tensor(scores):
                scores = scores.cpu().numpy()
            elif isinstance(scores, list):
                scores = torch.cat(scores, dim=0).cpu().numpy()
                
            mean_score = np.nanmean(scores)
            std_score = np.nanstd(scores)
            dice_metric.reset()
            metric_values.append(mean_score)
            metric_std_values.append(std_score)
            
            if mean_score > best_metric:
                best_metric = mean_score
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_hippocampus_unet.pth")
                print(f"   => New Best Validation Dice Score: {mean_score:.4f} ± {std_score:.4f} (Saved Model)")
            else:
                print(f"   => Validation Dice Score: {mean_score:.4f} ± {std_score:.4f}")

print(f"Training Complete. Best Metric: {best_metric:.4f} at epoch {best_metric_epoch}")""")

    add_markdown("## 6. Visualization & Clinical Output\nWe plot the loss curves, and more importantly, visualize the AI's predicted 3D segmentation mask against the real Ground Truth.")
    add_code("""# Plotting Training Curves
import numpy as np
epochs = np.arange(1, max_epochs + 1)
val_epochs = np.arange(val_interval, max_epochs + 1, val_interval)

plt.figure(figsize=(14, 5))

# Plot Training Loss with Standard Deviation
plt.subplot(1, 2, 1)
plt.title("Epoch Average Dice Loss (± 1 SD)")
plt.plot(epochs, epoch_loss_values, label="Train Loss", color="blue")
plt.fill_between(
    epochs, 
    np.array(epoch_loss_values) - np.array(epoch_loss_std_values), 
    np.array(epoch_loss_values) + np.array(epoch_loss_std_values), 
    color="blue", alpha=0.2
)
plt.xlabel("Epoch")
plt.legend()

# Plot Validation Dice with Standard Deviation
plt.subplot(1, 2, 2)
plt.title("Validation Mean Dice Score (± 1 SD)")
plt.plot(val_epochs, metric_values, label="Val Dice", color="green")
plt.fill_between(
    val_epochs, 
    np.array(metric_values) - np.array(metric_std_values), 
    np.array(metric_values) + np.array(metric_std_values), 
    color="green", alpha=0.2
)
plt.xlabel("Epoch")
plt.legend()
plt.show()""")

    add_code("""# Visualizing a validation sample
model.load_state_dict(torch.load("best_hippocampus_unet.pth"))
model.eval()

# Grab one validation volume
val_data = next(iter(val_loader))
val_inputs = val_data["image"].to(device)
val_labels = val_data["label"].to(device)

with torch.no_grad():
    val_outputs = sliding_window_inference(val_inputs, (32, 32, 32), 4, model)
    val_preds = torch.argmax(val_outputs, dim=1, keepdim=True)

# Select a 2D slice from the 3D volume that contains the hippocampus
image_volume = val_inputs[0, 0].cpu().numpy()
label_volume = val_labels[0, 0].cpu().numpy()
pred_volume = val_preds[0, 0].cpu().numpy()

# Find the slice with the maximum hippocampus pixels (to ensure we see it)
slice_idx = np.argmax(np.sum(label_volume, axis=(0, 1)))

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title("Raw MRI (T1)")
plt.imshow(image_volume[:, :, slice_idx], cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth Mask")
plt.imshow(image_volume[:, :, slice_idx], cmap="gray")
plt.imshow(label_volume[:, :, slice_idx], cmap="jet", alpha=0.5)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("AI Predicted Mask")
plt.imshow(image_volume[:, :, slice_idx], cmap="gray")
plt.imshow(pred_volume[:, :, slice_idx], cmap="jet", alpha=0.5)
plt.axis("off")
plt.show()

# Calculate clinical volume of the predicted Hippocampus
# (Assuming 1x1x1 mm voxel size for simplicity)
volume_mm3 = np.sum(pred_volume > 0)
print(f"Calculated Patient Hippocampal Volume: {volume_mm3} mm³")
print("This volume is used by neurologists to compare against healthy baselines for Alzheimer's diagnosis.")""")

    # Save to file
    with open('Kaggle_Hippocampus_Segmentation.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
        
    print("Successfully generated Kaggle_Hippocampus_Segmentation.ipynb!")

if __name__ == "__main__":
    create_notebook()
