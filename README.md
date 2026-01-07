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
