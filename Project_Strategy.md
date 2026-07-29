# Alzheimer's MRI Classification & Explainable AI (Grad-CAM) Project Strategy

## 1. The Historical Problem (The "Gap")
Early deep learning models for Alzheimer's classification achieved high accuracy but functioned as **"Black Boxes."** They could predict if a patient had Alzheimer's, but they could not physically show doctors *where* the brain shrinkage was occurring. 
* **Key Papers demonstrating this gap:** 
  * Korolev et al. (2017) - *Residual and plain convolutional neural networks for 3D brain MRI classification*
  * Payan & Montana (2015) - *Predicting Alzheimer's disease: a neuroimaging study with 3D convolutional neural networks*
* **The issue:** These models used computationally heavy raw 3D MRI scans from the ADNI dataset but failed to provide visual interpretability.

## 2. Our Proposed Solution
We will solve this "black box" gap by building an **Explainable AI (xAI)** model. We will use a 2D Convolutional Neural Network (CNN) combined with **Grad-CAM** (Gradient-weighted Class Activation Mapping). 
* Instead of just outputting a probability (e.g., "90% chance of Alzheimer's"), our model will draw a color-coded **heatmap** directly over the brain MRI slice.
* This heatmap will visually prove to clinicians exactly which pixels the AI is looking at to make its decision, turning the model into a "white box."
* **Supporting Modern Literature (Our Blueprint):**
  * Chattopadhyay et al. (2024) - *Comparison of Explainable AI Models for MRI-based Alzheimer’s Disease Classification*
  * *AXIAL (2024)* - *Attention-based eXplainability for Interpretable Alzheimer's Localized Diagnosis*

## 3. The Dataset Strategy
Training massive 3D MRIs requires supercomputers. To make this project highly feasible on standard PCs while retaining high accuracy, we will use **2D MRI slices**.
* **Instant Prototyping Dataset:** We will use the **Augmented Alzheimer MRI Dataset** from Kaggle.
  * **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)
  * **Details:** This contains over 33,000 pre-processed 2D MRI images divided into exactly 4 folders/classes: `Non_Demented`, `Very_Mild_Demented`, `Mild_Demented`, and `Moderate_Demented`.
* **The Source & Input Structure:** This Kaggle data is directly derived from the **ADNI** (Alzheimer's Disease Neuroimaging Initiative) database. The modern research papers (like AXIAL 2024) literally take the heavy 3D ADNI scans and slice them into 2D images, resizing them to a standard input shape of **224 x 224 x 3** (copying the grayscale MRI into 3 color channels). The Kaggle dataset has already done this exact slicing step for us, meaning our input structure perfectly matches state-of-the-art methodology.

## 4. How We Will Validate Our Model to Professors
Generating a heatmap isn't enough; we must prove to our professors that the heatmap is medically accurate and not just highlighting random background noise. We will validate our results using three methods:
1. **Medical Ground Truth:** We will prove that our Grad-CAM "red zones" (high attention areas) consistently highlight the **Hippocampus** (the memory center which shrinks in Alzheimer's) and the **Ventricles** (fluid gaps that expand as brain tissue dies).
2. **Literature Comparison:** We will place our generated heatmaps side-by-side with the published heatmaps from the 2024 research papers to prove our AI learned the exact same true pathological features as state-of-the-art models.
3. **The Sanity Check:** We will contrast the heatmap of a Healthy patient against an Alzheimer's patient to visually prove the AI recognizes the clear lack of brain atrophy in healthy individuals.
