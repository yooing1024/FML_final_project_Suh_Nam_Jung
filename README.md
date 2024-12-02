# Biomarker Analysis on OLIVES Dataset

## Overview

This repository contains the implementation of biomarker analysis using Optical Coherence Tomography (OCT) images from the OLIVES dataset. The project focuses on accurately detecting ophthalmic biomarkers through deep learning techniques, employing CNN-based (VGG16) and Transformer-based (DeiT) models. The analysis addresses challenges such as data scarcity, class imbalance, and multi-modal inputs. The results provide insights into the efficacy of combining convolutional and transformer paradigms for medical imaging applications, particularly in resource-constrained settings.

## Dataset

The OLIVES dataset comprises:

9396 pairs of grayscale OCT scans: Resolution of [496 x 504].
Clinical data: Best Corrected Visual Acuity (BCVA) and Central Subfield Thickness (CST).
Biomarkers: 16 binary labels.
Note: Data points with missing values were excluded.

## Key Features

Comparative Analysis: VGG16 (CNN) vs. DeiT (Transformer).
Data Imbalance Mitigation: Weighted BCE loss, focal loss, and optimized dataset splitting.
Multi-Modality: Integration of OCT image features and clinical data.
Pretrained Models: Utilization of transfer learning with custom classifier heads.
Experiment Setup

## Hardware
GPU: NVIDIA H100 GPU
## Hyperparameters
Epochs: 10
Optimizer: AdamW
Learning Rate: Experimented with 1e-3 and 1e-4 (optimal rate selected).
Scheduler: StepLR
Dataset Split: 60% training, 20% validation, 20% testing.
Pretrained Models
VGG16: Input size of [224 x 224].
DeiT: Evaluated with input sizes of [224 x 224] and [384 x 384].
## Methodology

### Custom Classifier Head
To adapt pretrained vision models for multi-label classification, we:
Added a single fully-connected (FC) layer or two FC layers with a dropout rate of 0.5 to reduce overfitting.
Retained pretrained weights for feature extraction.
### Loss Functions
Weighted BCE Loss: Addressed positive label scarcity.
Focal Loss: Parameters: γ = 2, α = 0.25, scaled by positive label frequency.
### Dataset Splitting
Random Split: Baseline strategy.
Eye-Wise Split: Ensures even distribution of positive labels across splits.
###  Multi-Modality
Feature Concatenation: Combined normalized clinical data with extracted OCT features.
Ensemble Learning: Separate models for clinical data (MLP) and OCT scans (vision models) combined via ensemble methods.

## Results
### Preliminary Experiments
Identified the optimal combination of loss functions and dataset splitting strategies.
Eye-wise split with focal loss performed best for vanilla baseline models.
### Main Experiments
VGG16 vs. DeiT: DeiT with [384 x 384] input size outperformed VGG16 in preserving OCT image details.
Multi-Modality: Combining OCT and clinical data significantly improved biomarker detection accuracy.
