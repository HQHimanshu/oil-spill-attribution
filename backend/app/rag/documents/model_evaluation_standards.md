# Machine Learning Model Evaluation & Confidence vs Accuracy Standards

**Source**: OceanGuard AI Engineering & Remote Sensing Machine Learning Group  
**Title**: Distinction Between Model Prediction Confidence and Empirical Validation Performance in SAR Segmentation  
**Date**: 2024-03-01  
**Document Type**: ML Engineering Standard  
**URL**: https://oceanguard.ai/docs/ml-evaluation-metrics  

## 1. Metrics Definitions
Segmentation performance is empirically evaluated on held-out ground truth test scenes using:
* **Intersection over Union (IoU / Jaccard Index)**: $IoU = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$. Measures the overlap proportion between predicted and ground truth oil masks.
* **Dice Coefficient (F1-Score)**: $Dice = \frac{2 |A \cap B|}{|A| + |B|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$. Balances precision and recall.
* **Precision**: $Precision = \frac{TP}{TP + FP}$. Minimizes false alarms from look-alikes.
* **Recall**: $Recall = \frac{TP}{TP + FN}$. Minimizes missed slick detections.

## 2. Distinction: Prediction Confidence vs Validation Performance
* **Validation Performance (IoU: 98.65%, Dice: 99.32%)**: An objective, benchmarked metric derived by testing the trained model on labeled ground-truth scenes.
* **Prediction Confidence (e.g., 94%)**: An inference-specific certainty estimation calculated for an individual SAR image based on local backscatter damping gradient and feature distance. High confidence does not guarantee ground-truth accuracy in uncharted conditions.
