# 🌌 Exoplanet Classification Using Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Colab](https://img.shields.io/badge/Google-Colab-F9AB00? logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Advanced Deep Learning system for classifying NASA Kepler exoplanet candidates with 91.33% F1-Score, outperforming Bayesian baselines by 16.84%**

---

## 📝 Executive Summary

This project addresses the critical challenge of distinguishing genuine exoplanets from false positives in NASA's Kepler mission data. We developed and optimized multiple Neural Network architectures, comparing them against a Bayesian baseline (Naive Bayes) using professional MLflow experiment tracking.

**Key Results:**
- 🏆 **Best Model**: 4-layer Deep ReLU architecture achieving 91.33% F1-Score
- 📊 **Dataset**: 7,585 Kepler candidates with 15 astrophysical features
- 🎯 **Impact**: 16.84% improvement over Naive Bayes, validating Deep Learning approach
- ⚡ **Balance**: Optimal precision (90.12%) and recall (92.58%) for scientific discovery

**Methodology**:  Comprehensive pipeline including data preprocessing, architecture optimization (4 neural networks + Bayesian baseline), hyperparameter tuning, and rigorous evaluation with MLflow tracking for reproducibility.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Models](#-models)
- [Results](#-results)
- [Quick Start](#-quick-start)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Authors](#-authors)

---

## 🎯 Problem Statement

### Challenge
NASA's Kepler mission identified thousands of exoplanet candidates, but not all are genuine planets—many are false positives from stellar activity or instrumental noise.  Accurate classification is crucial for optimizing telescope resources and accelerating discovery. 

### Solution
Binary classification system distinguishing:
- ✅ **CONFIRMED**: Validated exoplanets (36.2% of data)
- ❌ **FALSE POSITIVE**:  Erroneous detections (63.8% of data)

### Impact
- **Scientific**: Prioritizes real candidates for follow-up observations
- **Economic**: Saves expensive telescope time
- **Discovery**:  Accelerates exoplanet science

---

## 📊 Dataset

**Source**: [NASA Exoplanet Archive - Kepler Cumulative Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)

| Metric | Value |
|--------|-------|
| **Total Objects** | 9,564 KOI (Kepler Objects of Interest) |
| **Features** | 83 → 15 selected |
| **Final Dataset** | 7,585 (after filtering CANDIDATE class) |
| **Train/Test Split** | 80% / 20% (stratified) |
| **Class Imbalance** | 1.76:1 (FALSE POSITIVE :  CONFIRMED) |

### Selected Features (15)

**Planetary**: orbital period, radius, temperature, insolation  
**Transit**: duration, depth, impact parameter, SNR  
**Stellar**: effective temperature, surface gravity, radius  
**Observational**: Kepler magnitude, sky coordinates, epoch

*Feature selection criteria: <30% missing values, astrophysical relevance, discriminative power*

---

## 🔬 Methodology

### Pipeline

### Key Steps

1. **Preprocessing**
   - Removed CANDIDATE class (unconfirmed objects)
   - Eliminated features with >30% missing values
   - StandardScaler normalization (μ=0, σ=1)

2. **Model Development**
   - Baseline: 2-layer simple network
   - Optimization: 3 architectures with varying depth/activation
   - Bayesian baseline: Gaussian Naive Bayes

3. **Hyperparameter Tuning**
   - Optimizers: Adam vs RMSprop
   - Learning rates: 0.001, 0.0005
   - Dropout: 0.2-0.3
   - Early stopping (patience=10)

4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - MLflow tracking for all experiments
   - Statistical significance testing

---

## 🧠 Models

### Neural Networks

| Model | Architecture | Activation | F1-Score | Notes |
|-------|-------------|------------|----------|-------|
| **Model_2** 🏆 | 4 layers (32→16→8→1) | ReLU + Sigmoid | **0.9133** | Deep, dropout 30-20% |
| Model_3 | 3 layers (16→8→1) | ReLU + Sigmoid | 0.9025 | RMSprop optimizer |
| Model_1 | 3 layers (16→8→1) | Full Sigmoid | 0.8884 | Sigmoid comparison |
| Baseline | 2 layers (4→1) | Sigmoid | 0.8118 | Initial benchmark |

### Bayesian Baseline

| Model | Type | F1-Score | Characteristics |
|-------|------|----------|-----------------|
| Naive Bayes | Gaussian | 0.7381 | High precision (98.36%), low recall (59.06%) |

**Key Differences**:
- **ReLU**: Prevents vanishing gradients, faster convergence
- **Dropout**: Regularization to prevent overfitting
- **Adam**:  Adaptive learning rates outperform RMSprop

---

## 📈 Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Improvement vs NB |
|-------|----------|-----------|--------|----------|-------------------|
| **Model_2** | 89.01% | 90.12% | 92.58% | **0.9133** | **+23.7%** |
| Model_3 | 87.24% | 91.66% | 87.55% | 0.9025 | +22.3% |
| Model_1 | 86.28% | 92.71% | 84.72% | 0.8884 | +20.4% |
| Baseline | 76.04% | 78.17% | 85.59% | 0.8118 | +9.9% |
| Naive Bayes | 73.79% | 98.36% | 59.06% | 0.7381 | — |

### Model_2 Confusion Matrix (Test Set)
            Predicted
          CONFIRMED  FALSE POSITIVE
  Actual CONFIRMED 538 11 (98% correct) FALSE POSITIVE 68 848 (93% correct)

**Key Metrics**:
- **False Negative Rate**: 2% (rarely misses real planets)
- **False Positive Rate**: 7.4% (acceptable for follow-up)
- **Prediction Confidence**: 85% of predictions >90% confidence

---

## 🚀 Quick Start

### Google Colab (Recommended)

1. **Open Notebook**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SMachadoP/Exoplanet-Classification-Using-Neural-Networks/blob/main/proyecto_exoplanetas.ipynb)

2. **Run All Cells**: `Runtime` → `Run all` (~45 min execution)

3. **View Results**:  All visualizations and metrics generated inline

### Local Setup

```bash
git clone https://github.com/SMachadoP/Exoplanet-Classification-Using-Neural-Networks.git
cd Exoplanet-Classification-Using-Neural-Networks
pip install -r requirements.txt
jupyter notebook proyecto_exoplanetas.ipynb

### MLflow UI (Local Only)
mlflow ui  # Access at http://localhost:5000
For Colab users: Download MLflow artifacts and run locally.

🔍 Key Findings
1. Deep Learning Justification ✅
Neural networks outperformed Naive Bayes by 16.84% in F1-Score (p < 0.0001), validating the use of complex models for this task.

Why:

Capture non-linear feature interactions
No independence assumption (unlike Naive Bayes)
Learn hierarchical representations
2. Optimal Architecture 🏆
Model_2 (4-layer Deep ReLU) achieved best results through:

Progressive depth (32→16→8→1 neurons)
ReLU activation (solves vanishing gradients)
Dropout regularization (30%→30%→20%)
Adam optimizer (adaptive learning)
3. Precision-Recall Balance ⚖️
Naive Bayes: Conservative (98% precision, 59% recall) → Misses many real planets Model_2: Balanced (90% precision, 93% recall) → Optimal for discovery ✅

4. Feature Importance 📊
Top discriminative features: Signal-to-noise ratio, transit depth, transit duration

5. Model Confidence 🎯
85% of predictions have >90% confidence with 94% accuracy → Reliable for scientific decisions
💻 Technologies
Technology	Purpose
TensorFlow/Keras	Neural network framework
Scikit-learn	Preprocessing, Naive Bayes, metrics
MLflow	Experiment tracking & model registry
Pandas/NumPy	Data manipulation
Matplotlib/Seaborn	Visualization
ML Techniques: Binary classification, deep learning, Bayesian inference, hyperparameter optimization, dropout regularization, early stopping

👥 Authors
Sebastián Machado & Sebastián Verdugo

Course: Artificial Intelligence
Institution: Universidad Del Desarrollo
Date: December 3, 2025
GitHub: @SMachadoP
🔮 Future Work
Class imbalance: SMOTE oversampling
Advanced architectures: CNNs, attention mechanisms
Ensemble methods: Stacking top models
Explainability: SHAP values, LIME
Deployment: REST API for real-time predictions

@software{machado2025exoplanet,
  author = {Machado, Sebastián and Verdugo, Sebastián},
  title = {Exoplanet Classification Using Neural Networks},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/SMachadoP/Exoplanet-Classification-Using-Neural-Networks}
}
