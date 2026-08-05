# Model 5: 3D Semantic Segmentation of the Hippocampus

## Overview
This repository contains the code and deployment notebooks for **Model 5**, a True 3D Semantic Segmentation pipeline aimed at the early detection of Alzheimer's Disease. 

While earlier iterations of our project focused on weakly-supervised classification (which could only highlight the edges of brain atrophy), this model performs **pixel-perfect 3D boundary segmentation** of the Hippocampus. The Hippocampus is typically the first brain structure to atrophy during the onset of Alzheimer's. By isolating this structure, we can mathematically calculate the exact Hippocampal Volume in $mm^3$ and compare it against healthy baselines—the clinical gold standard for identifying Mild Cognitive Impairment (MCI) and early-stage Alzheimer's.

## Dataset
We transitioned our data pipeline to utilize the **Medical Segmentation Decathlon (MSD) - Task 04 (Hippocampus)** dataset.
- **Format**: True 3D MRI scans (`.nii.gz` NIfTI format).
- **Volume**: 394 high-resolution scans.
- **Ground Truth**: Each scan is paired with an expert-traced ground-truth mask of the anterior and posterior hippocampus. This allows the AI to learn precise boundary mapping.

## Preprocessing Pipeline
Given the massive computational requirements of 3D MRI data, we leveraged **MONAI** (Medical Open Network for AI) to handle our data pipelines efficiently, bypassing Kaggle's GPU VRAM limits.
Our preprocessing dictionary transforms include:
1. `LoadImaged`: Efficiently loads the 3D `.nii.gz` volumes.
2. `CropForegroundd`: Removes empty background space around the brain to reduce memory overhead.
3. `RandCropByPosNegLabeld`: Extracts randomized 3D patches from the larger volume during training, ensuring the model focuses on both the hippocampus and background equally without exceeding VRAM limits.

## Architecture
The core architecture is a **3D U-Net**, implemented via MONAI. 
- The U-Net structure is heavily optimized for biomedical image segmentation, featuring a contracting encoder path to capture context and a symmetric expanding decoder path that enables precise localization.
- It operates entirely in 3D, allowing it to understand depth and spatial relationships across multiple MRI slices simultaneously.

## Training & Metrics
- **Loss Function:** We transitioned from CrossEntropy to **Dice Loss**. Dice Loss (closely related to Intersection over Union - IoU) directly penalizes the model for drawing boundaries that do not perfectly overlap with the doctor's expert annotations.
- **Environment:** The deployment notebook (`Kaggle_Hippocampus_Segmentation.ipynb`) is pre-configured to run efficiently on Kaggle's P100/T4 GPUs.

## Repository Contents
* `Kaggle_Hippocampus_Segmentation.ipynb`: The primary executable notebook for training the 3D U-Net.
* `build_segmentation_notebook.py`: The script utilized to generate the Kaggle notebook.
* `predict_3d.py` & `evaluate_oasis.py`: Post-training scripts used for evaluating the model on test volumes and rendering 3D visualizations.
* `best_hippocampus_unet.pth`: The saved weights of our best-performing trained model.

*(Note: The heavy NIfTI dataset files and massive 3D HTML visualizations have been excluded from this repository to save space and comply with GitHub's file limits.)*
