# COVID-19 Detection from Chest X-Ray Images Using GBM with Comparative Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: 【entity-MIT¦canonical_name=MIT】](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of our published research paper: **"COVID-19 Detection from Chest X-Ray Images Using GBM with Comparative Analysis"** - published in MIND 2023, CCIS 2128, Springer.

**Authors:** Abisek Dahal, Abu Motaleb Rony, and Soumen Moulik  
**Affiliation:** Department of CSE, National Institute of Technology, Shillong, India

> Paper Link: https://doi.org/10.1007/978-3-031-62217-5_20

### 📖 Abstract
Effective disease management depends on quick and precise diagnosis of COVID-19. This study presents a Gradient Boosting Machines (GBM) and Convolutional Neural Networks (CNNs)-based model for COVID-19 detection, along with a comprehensive comparative analysis of various other machine learning (ML) algorithms. Our GBM and 【entity-CNN¦canonical_name=CNN】 model successfully differentiates between COVID-19-positive cases and healthy people with a remarkable accuracy of **97.41%**, but GBM takes less computation time compared to 【entity-CNN¦canonical_name=CNN】.

### 📊 Dataset
We used a publicly available dataset from Kaggle:
**Source:** [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

- **Total Images:** 2905 Chest X-Ray images
- **COVID-19 Cases:** 219
- **Normal Cases:** 1341
- **Viral Pneumonia Cases:** 1345
- **Split Ratio:** 80% Training, 20% Testing

**Preprocessing:**
- Resizing to 224x224 pixels
- Normalization (Min-Max & Z-score)
- Data Augmentation (Rotation, Zoom, Horizontal Flip, Shift)
- Noise Reduction

### 🧠 Algorithms Compared
We evaluated 11 algorithms on the same dataset:
1.  【entity-CNN¦canonical_name=CNN】 - Convolutional Neural Network
2.  GBM - Gradient Boosting Machine **(Best Performer)**
3.  XGBoost - eXtreme Gradient Boosting
4.  SVM - Support Vector Machine
5.  K-NN - K-Nearest Neighbors
6.  Decision Tree
7.  Naive Bayes
8.  Logistic Regression
9.  Random Forest
10. RNN - Recurrent Neural Network
11. AdaBoost

### 🏆 Results

#### Performance Metrics
| Algorithm | Precision (P) | Recall (R) | F1-Score | Specificity (S) | Accuracy (A) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **【entity-CNN¦canonical_name=CNN】** | 1 | 0.375 | 0.545 | 1 | **0.974** |
| **GBM** | 1 | 0.370 | 0.540 | 1 | **0.974** |
| K-NN | 0.8 | 0.5 | 0.615 | 0.994 | 0.974 |
| SVM | 0.5 | 0.625 | 0.555 | 0.972 | 0.958 |
| RNN | 0.98 | 0.378 | 0.6 | 1 | 0.958 |

#### Computation Time (In Seconds)
| Algorithm | Training Time | Testing Time | Total Time |
| :--- | :--- | :--- | :--- |
| **【entity-CNN¦canonical_name=CNN】** | 980.25 | 60.23 | **1040.48** |
| **GBM** | 420.12 | 32.12 | **452.24** |
| SVM | 870.49 | 20.95 | 891.44 |
| KNN | 70.85 | 12.18 | 83.03 |

**Key Finding:** GBM and 【entity-CNN¦canonical_name=CNN】 both achieved 97.4% accuracy, but GBM is ~2.3x faster than 【entity-CNN¦canonical_name=CNN】, making it more suitable for clinical deployment.

### 📁 Repository Structure
