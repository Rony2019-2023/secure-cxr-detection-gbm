# COVID-19 Detection from Chest X-Ray Images: GBM and 10 Other Models (Original vs. Updated on 7,001 Images)

Comparative evaluation of 11 machine learning models for binary COVID-19 detection from chest X-ray (CXR) images on an expanded dataset of 7,001 images (COVID vs non-COVID, where non-COVID = Normal + Viral pneumonia), reporting sensitivity, specificity, F1, ROC-AUC and PR-AUC instead of accuracy alone. This repository contrasts the original published findings on 2,905 images with an expanded, robust experiment, addressing previous limitations such as severe class imbalance, tiny test sets, and low COVID recall, with a long-term vision of building Trustworthy and Robust Medical AI systems.

## Publication

**COVID-19 Detection from Chest X-Ray Images Using GBM with Comparative Analysis**  
Abisek Dahal, Abu Motaleb Rony, Soumen Moulik  
Springer CCIS 2128 (MIND 2023, published 2024), Scopus indexed  
DOI: 10.1007/978-3-031-62217-5_20

In the original paper, both GBM and CNN achieved 97.41% accuracy. However, GBM required less computation time than CNN.

This repository: An updated pipeline on 7,001 CXR images with binary labels. The focus is COVID sensitivity and balanced metrics, not headline accuracy alone.

**The paper's numbers and these numbers are not directly comparable. The datasets, class balance and test sets differ (see the comparison below).**

### 1. Task Setup & Implementation Details

| Item | Detail |
| :--- | :--- |
| Task | Binary classification |
| COVID (1) | COVID-19 positive CXR |
| Non-COVID (0) | Normal + Viral pneumonia, merged into one class |
| Dataset | COVID-19 Radiography Database (Kaggle) |
| Images | 7,001: COVID-19 3,618, Normal 2,038, Viral pneumonia 1,345 (non-COVID total 3,383) |
| Split | 80% train / 20% test, stratified, random_state=42 (5,600 / 1,401) |
| Test set | 724 COVID-19, 677 non-COVID (399 Normal, 278 Viral pneumonia) |
| Input | 224 x 224, grayscale, scaled to [0, 1] |


### 📁 Dataset Organization

The dataset was downloaded from Kaggle(https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and stored in MyDrive. As the data set was very big I used 3618 images of covid, 2038 images of Normal, 1345 of viral Pneumonia.

**Folder Structure:**
```bash
/MyDrive/FullData/
├── COVID/      # COVID-19 positive cases (3,618 images)
├── Normal/     # Normal chest X-rays (2,038 images)
└── Viral/      # Viral Pneumonia cases (1,345 images)
```

**Binary Labeling:**
- `Non-COVID (0)` = Normal + Viral
- `COVID (1)` = COVID folder

**Labeling Rule:**
- **Label 0 (Non-COVID):** Every image from the Normal and Viral Pneumonia folders was labeled as 0.
- **Label 1 (COVID):** Every image from the COVID-19 folder was labeled as 1.

This binary labeling strategy aligns with a practical clinical screening goal: distinguishing COVID-19 positive cases from all other cases (COVID vs. Everything Else).

**Labeling rule:** Every Normal and Viral pneumonia image is labeled 0, every COVID-19 image is labeled 1. This matches a practical screening goal: COVID vs everything else.

### 2. Limitations Removed & What Changed

| Limitation in the Published Paper | Status in This Updated Pipeline |
| :--- | :--- |
| Very small test set | Fixed: Expanded test set to 1,401 images |
| Severe class imbalance | Addressed: Nearly balanced dataset (~51.7% COVID) |
| Very low COVID recall (0.25 to 0.63) | Improved: Recall 0.95–0.98 |
| Misleading high accuracy | Addressed: Reported F1, Specificity, ROC-AUC, PR-AUC |
| Sensitivity formula error Eq. (6) | Corrected: Sen = TP / (TP+FN) |

### 3. Original Paper Results (2,905 Images)

| Algorithm | Precision | Recall | F1-Score | Specificity | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CNN | 1.000 | 0.375 | 0.545 | 1.000 | 0.974 |
| SVM | 0.500 | 0.625 | 0.555 | 0.972 | 0.958 |
| K-NN | 0.800 | 0.500 | 0.615 | 0.994 | 0.974 |
| Decision Tree | 0.166 | 0.375 | 0.230 | 0.918 | 0.896 |
| Naive Bayes | 0.161 | 0.625 | 0.256 | 0.859 | 0.849 |
| Logistic Regression | 0.444 | 0.500 | 0.470 | 0.972 | 0.953 |
| Random Forest | 1.000 | 0.250 | 0.400 | 1.000 | 0.968 |
| RNN | 0.980 | 0.378 | 0.600 | 1.000 | 0.958 |
| AdaBoost | 0.500 | 0.250 | 0.333 | 0.989 | 0.958 |
| XGBoost | 1.000 | 0.250 | 0.400 | 1.000 | 0.968 |
| GBM | 1.000 | 0.370 | 0.540 | 1.000 | 0.974 |

### 4. Updated Results (7,001 Images, 1,401 Test Images)

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CNN | 0.964 | 0.952 | 0.981 | 0.947 | 0.966 | 0.991 |
| XGBoost | 0.936 | 0.927 | 0.950 | 0.920 | 0.939 | 0.984 |
| GBM | 0.934 | 0.923 | 0.950 | 0.916 | 0.937 | 0.983 |
| SVM | 0.931 | 0.899 | 0.975 | 0.883 | 0.936 | 0.979 |
| KNN | 0.928 | 0.901 | 0.967 | 0.886 | 0.933 | 0.968 |
| Random Forest | 0.926 | 0.906 | 0.957 | 0.894 | 0.931 | 0.980 |
| Logistic Regression | 0.882 | 0.877 | 0.896 | 0.866 | 0.887 | 0.942 |
| Decision Tree | 0.865 | 0.860 | 0.883 | 0.846 | 0.871 | 0.839 |
| AdaBoost | 0.847 | 0.836 | 0.876 | 0.817 | 0.856 | 0.923 |
| Naive Bayes | 0.841 | 0.813 | 0.899 | 0.778 | 0.854 | 0.844 |
| RNN | 0.742 | 0.776 | 0.703 | 0.783 | 0.738 | 0.803 |

### 5. Future Work: Towards a Trustworthy & Robust Medical AI

- Adversarial Robustness & Cybersecurity for AI (ART)
- Federated Learning for Healthcare (Flower)
- Explainable AI (XAI): Grad-CAM and SHAP
- Rigorous Validation: Stratified k-fold, patient-level splitting
- Advanced Feature Extraction: ResNet, EfficientNet, PCA

### Authors

- Abisek Dahal, NIT Meghalaya
- Abu Motaleb Rony, NIT Meghalaya (B.Tech CSE); Instructor, NSTI Nilphamari
    - Email: abumotalebrony@gmail.com
    - GitHub: github.com/Rony2019-2023/secure-cxr-detection-gbm
- Soumen Moulik, NIT Meghalaya

### Citation

@inproceedings{dahal2024covid,
  title = {COVID-19 Detection from Chest X-Ray Images Using GBM with Comparative Analysis},
  author = {Dahal, Abisek and Rony, Abu Motaleb and Moulik, Soumen},
  booktitle = {Machine Learning, Image Processing, Network Security and Data Sciences (MIND 2023)},
  series = {Communications in Computer and Information Science},
  volume = {2128},
  pages = {239--247},
  year = {2024},
  publisher = {Springer, Cham},
  doi = {10.1007/978-3-031-62217-5_20}
}

### Disclaimer

Research and educational use only. Not a medical device. Do not use as the sole basis for clinical diagnosis.
