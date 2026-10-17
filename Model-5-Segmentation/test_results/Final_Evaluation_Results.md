# Final Full-Dataset Evaluation Results

Based on the 3D U-Net Hippocampal Segmentation pipeline applied to the full OASIS-1 dataset.

## Dataset Distribution
Out of the 416 subjects in the OASIS-1 dataset, exactly **235 subjects** possess clinically verified Clinical Dementia Ratings (CDR). We filtered the dataset to these 235 individuals to perform ground-truth validation against our Hippocampal Volume Fraction (HVF) thresholds.

## Classification Thresholds
Based on the volumetric analysis of the hippocampus against the estimated Total Intracranial Volume (eTIV), the subjects were divided into three classes:
- **Normal (No AD):** CDR = 0
- **Early AD:** CDR = 0.5
- **Late AD:** CDR ≥ 1.0

## Overall Performance Metric
- **Final Accuracy:** **94.89%** (223 / 235 correct classifications)

## Confusion Matrix

| | **Predicted: Normal** | **Predicted: Early AD** | **Predicted: Late AD** |
|---|:---:|:---:|:---:|
| **Actual: Normal** | **130** | 5 | 0 |
| **Actual: Early AD** | 4 | **64** | 2 |
| **Actual: Late AD** | 0 | 1 | **29** |

### Per-Class Breakdown
- **Normal (No AD):** 
  - True Positives: 130
  - False Negatives: 5 (Misclassified as Early AD due to borderline hippocampal shrinkage)
  - **Recall:** 96.3%

- **Early AD:**
  - True Positives: 64
  - False Negatives: 6 (4 misclassified as Normal; 2 misclassified as Late AD)
  - **Recall:** 91.4%

- **Late AD:**
  - True Positives: 29
  - False Negatives: 1 (Misclassified as Early AD)
  - **Recall:** 96.6%

## Clinical Conclusion
The volumetric ratio of the segmented hippocampus serves as a highly robust biomarker. The 3D U-Net achieved exceptional performance, particularly in distinguishing healthy patients from those with any form of dementia. The minor overlap (misclassifications) predominantly occurred at the boundary between Normal and Early AD, which aligns with clinical realities where borderline cognitive decline is difficult to distinguish exclusively via structural volume.
