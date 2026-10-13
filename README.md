# COVID-19 Detection from Chest X-Ray Images: GBM and 10 Other Models, Revisited on 7,001 Images

Comparative evaluation of 11 machine learning models for binary COVID-19 detection from chest X-ray (CXR) images on an expanded dataset of 7,001 images (COVID vs non-COVID, where non-COVID = Normal + Viral pneumonia), reporting sensitivity, specificity, F1, ROC-AUC and PR-AUC instead of accuracy alone.

## Publication

**COVID-19 Detection from Chest X-Ray Images Using GBM with Comparative Analysis**
Abisek Dahal, Abu Motaleb Rony, Soumen Moulik
Springer CCIS 2128 (MIND 2023, published 2024), Scopus indexed
DOI: [10.1007/978-3-031-62217-5_20](https://doi.org/10.1007/978-3-031-62217-5_20)

**Original paper:** GBM and CNN reached 97.41% accuracy; GBM needed less computation time than CNN.

**This repository:** an updated pipeline on 7,001 CXR images with binary labels. The focus is COVID sensitivity and balanced metrics, not headline accuracy alone.

> **The paper's numbers and these numbers are not directly comparable.** The datasets, class balance and test sets differ (see the comparison below).

## Task setup

| Item | Detail |
| :--- | :--- |
| Task | Binary classification |
| COVID (1) | COVID-19 positive CXR |
| Non-COVID (0) | Normal + Viral pneumonia, merged into one class |
| Dataset | [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), three class folders in a `FullData` directory *(add the download date or version)* |
| Images | 7,001: COVID-19 3,618, Normal 2,038, Viral pneumonia 1,345 (non-COVID total 3,383) |
| Split | 80% train / 20% test, stratified, `random_state=42` (5,600 / 1,401) |
| Test set | 724 COVID-19, 677 non-COVID (399 Normal, 278 Viral pneumonia) |
| Input | 224 x 224, grayscale, scaled to [0, 1] |

**Image selection.** All COVID-19 and Viral pneumonia images were used. For the Normal class, the **first 2,038 images by file number** were taken (about `Normal-1` to `Normal-2038`). This is not a random sample and its effect was not tested.

Labeling rule: every Normal and Viral pneumonia image is labeled 0, every COVID-19 image is labeled 1. This matches a practical screening goal: COVID vs everything else.

## Models compared (11)

| # | Model | Type | Class-balance handling |
| :--- | :--- | :--- | :--- |
| 1 | GBM | Ensemble (boosting) | `sample_weight` (balanced) |
| 2 | XGBoost | Ensemble (boosting) | `sample_weight` (balanced) |
| 3 | Random Forest | Ensemble (bagging) | `class_weight="balanced"` |
| 4 | AdaBoost | Ensemble (boosting) | `sample_weight` (balanced) |
| 5 | Decision Tree | Single tree | `sample_weight` (balanced) |
| 6 | SVM (RBF) | Kernel method | `class_weight="balanced"` |
| 7 | Logistic Regression | Linear | `class_weight="balanced"` + StandardScaler |
| 8 | KNN | Instance-based | Distance weights only (no class weights) |
| 9 | Naive Bayes | Probabilistic | Not applied in the current run |
| 10 | CNN | Deep learning | balanced class weights applied |
| 11 | RNN (SimpleRNN) | Deep learning | `sample_weight` (balanced) |

Classical models use flattened 224 x 224 pixels (50,176 features). The CNN uses input shape (N, 224, 224, 1) and was trained for 5 epochs with batch size 16. The RNN uses a sequence of shape (N, 50176, 1) as a baseline.

The two classes are nearly balanced (51.7% COVID in the test set, 51.7% in the whole dataset), so the class weights change little. Their effect was not measured separately.

## Updated results (7,001 images, 1,401 test images)

| Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1 | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN** | **0.964** | **0.952** | **0.981** | **0.947** | **0.966** | **0.991** |
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

- **Most accurate and most sensitive:** CNN (accuracy 0.964, recall 0.981, only 14 missed COVID images).
- **Best of the classical models:** XGBoost and GBM (they are indistinguishable here); SVM (0.975) and KNN (0.967) have higher recall but lower specificity.
- **Statistical comparison** (from the saved test predictions; Wilson 95% intervals and exact McNemar tests on paired predictions):

| Model | Accuracy (95% CI) | Sensitivity (95% CI) |
| :--- | :--- | :--- |
| CNN | 0.964 (0.953-0.973) | 0.981 (0.968-0.988) |
| XGBoost | 0.936 (0.922-0.947) | 0.950 (0.932-0.964) |
| GBM | 0.934 (0.919-0.946) | 0.950 (0.932-0.964) |

CNN vs GBM: 70 vs 27 images only one is right (p = 1.5e-5). CNN vs XGBoost: 66 vs 26 (p = 3.7e-5). GBM vs XGBoost: 16 vs 19 (p = 0.74).

## Original paper vs this pipeline

| Aspect | Original paper | This pipeline |
| :--- | :--- | :--- |
| Dataset size | 2,905 images (as reported) | 7,001 images |
| Test set | Very small: reported recalls are all multiples of 1/8 and 97.41% equals 188/193, consistent with about 8 COVID test images | 1,401 images (724 COVID, 677 non-COVID) |
| Class balance | Strongly imbalanced | Nearly balanced (51.7% COVID) |
| Non-COVID class | Normal (the SVM notebook of the paper code reads only the COVID-19 and Normal folders) | Normal + Viral pneumonia |
| Split | Random split, not stratified | Stratified, `random_state=42` |
| Headline accuracy | 97.4% (CNN and GBM) | 84% to 96% across models |
| COVID recall | 0.37 to 0.63 for the models in the original results table | 0.95 to 0.98 for the top models |
| Metrics | Precision, recall, F1, specificity, accuracy | Also ROC-AUC, PR-AUC, confusion matrices, saved predictions and metrics per model |
| Best model | CNN and GBM (equal accuracy) | CNN, clearly ahead of GBM and XGBoost on this test set |
| Interpretation | High accuracy on a tiny, imbalanced test set can hide missed COVID cases | Balanced test set, so accuracy and recall are both informative |

The higher recall mainly reflects a larger, nearly balanced dataset and a much larger test set. It should not be credited to class weights alone, because the effect of class weights was not isolated.

## Computation time (approximate, Google Colab)

| Model | Training (s) | Testing (s) | Notes |
| :--- | :--- | :--- | :--- |
| Naive Bayes | 1.4 | 2.0 | Fastest |
| KNN | 4.5 | 79 | Slow at inference |
| Decision Tree | 253 | 0.07 | |
| Random Forest | 106 | 0.2 | |
| Logistic Regression | 106 | 1.3 | |
| XGBoost | 504 | 0.7 | Strong accuracy and time trade-off |
| AdaBoost | 2,577 | 8.7 | |
| SVM | 2,838 | 480 | Slow on raw high-dimensional pixels |
| RNN | 5,638 | 97 | Long-sequence baseline |
| GBM | 6,727 | not recorded | Heavy on 50k features |
| CNN | 3,083 | 46 | Hardware used was not recorded |

The hardware assigned by Colab (CPU or GPU) was not recorded, so these times are indicative only and are **not a fair speed comparison across model types**. In particular, the paper's claim that GBM is faster than CNN was not reproduced: here the CNN trained faster than GBM (3,083 s vs 6,727 s), but the hardware may differ.

## Paper limitations: what changed

| Limitation in the published paper | Status in this repository |
| :--- | :--- |
| Very low COVID recall | Improved: recall 0.95 to 0.98 for the top models, mainly due to the larger, nearly balanced dataset |
| Misleading high accuracy on an imbalanced, tiny test set | Addressed: balanced test set (majority baseline 51.7%), F1, ROC-AUC and PR-AUC reported |
| Very small test set (about 8 COVID images) | Fixed: 724 COVID images in the test set, with confidence intervals |
| Binary vs three-class labeling unclear | Fixed: non-COVID = Normal + Viral pneumonia, COVID = COVID only |
| Small dataset (2,905) | Improved: 7,001 images |
| Sensitivity formula in Eq. (6) | See the note below |
| KNN and Naive Bayes without class weights | Still open (CNN now uses balanced class weights) |
| Single 80/20 split, no cross-validation | Still open |
| Duplicate or patient-level leakage not checked | Still open |
| Possible source bias (COVID and non-COVID images come from different collections) | Still open; no control experiment |
| Normal images selected as the first 2,038 by file number | New caveat; effect not tested |
| Data augmentation described in the paper | Not used in these notebooks |
| Raw 50k-dimension pixels for classical models | Still open (PCA or CNN features planned) |
| No external validation, no explainability, no robustness tests | Still open |

### Note on Eq. (6) of the paper

Eq. (6), labeled "Sensitivity", is written as

$$\text{Sen} = \frac{TP}{TP + FP}$$

which is Precision. The correct definition is

$$\text{Sensitivity (Recall)} = \frac{TP}{TP + FN}$$

Eq. (1) (Precision) and Eq. (2) (Recall) are correct. All metrics in this repository use the correct definitions.

## Evaluation metrics

| Metric | Formula | Meaning |
| :--- | :--- | :--- |
| Precision | TP / (TP + FP) | Of predicted COVID, how many are truly COVID |
| Recall / Sensitivity | TP / (TP + FN) | Of true COVID cases, how many the model finds |
| Specificity | TN / (TN + FP) | Of true non-COVID, how many are correctly rejected |
| F1-score | 2 x P x R / (P + R) | Harmonic mean of precision and recall |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correct predictions |

Also reported: ROC-AUC and PR-AUC (Average Precision).

## How to run

Requirements: Python 3.8+, numpy, pandas, matplotlib, pillow, scikit-learn, xgboost, tensorflow (CNN and RNN).

```bash
pip install numpy pandas matplotlib pillow scikit-learn xgboost tensorflow
```

1. Download the dataset from Kaggle and place the class folders (`Covid-19`, `Normal`, `Viral pneumonia`) in one directory (for example `Project/Mydataset/FullData/`).
2. Set `DATA_DIR` (and the results path) at the top of each notebook.
3. Labeling in code: Normal and Viral pneumonia map to 0, COVID-19 maps to 1. For Normal, use the first 2,038 images in sorted file order to reproduce the reported subset.
4. Open any of the 11 notebooks in Google Colab or Jupyter and run the cells top to bottom.
5. Metrics and predictions are saved under `results/` as `*_metrics.csv` and `*_test_predictions.csv`.

## Repository structure

```text
.
├── notebooks/        # 11 model notebooks
├── results/          # metrics and test predictions (CSV)
├── requirements.txt
├── LICENSE
└── README.md
```

## Future work

- [ ] Stratified k-fold cross-validation (mean and standard deviation)
- [ ] Duplicate image check and a patient-level split where metadata allows
- [ ] Control for the Normal selection and for source bias (random Normal sample, border-only classifier)
- [ ] Class weights for KNN and Naive Bayes
- [ ] PCA or CNN features for the classical models
- [ ] Transfer-learning baseline (ResNet or EfficientNet)
- [ ] External or multi-center validation
- [ ] Explainability (Grad-CAM, SHAP)
- [ ] Adversarial robustness evaluation

## Authors

- Abisek Dahal, NIT Meghalaya
- Abu Motaleb Rony, NIT Meghalaya (B.Tech CSE); Instructor, NIST Nilphamari
  - Email: motaleb.rony.phd@gmail.com
  - GitHub: motaleb-roni, Rony2019-2023
- Soumen Moulik, NIT Meghalaya

## Citation

```bibtex
@inproceedings{dahal2024covid,
  title     = {COVID-19 Detection from Chest X-Ray Images Using GBM with Comparative Analysis},
  author    = {Dahal, Abisek and Rony, Abu Motaleb and Moulik, Soumen},
  booktitle = {Machine Learning, Image Processing, Network Security and Data Sciences (MIND 2023)},
  series    = {Communications in Computer and Information Science},
  volume    = {2128},
  pages     = {239--247},
  year      = {2024},
  publisher = {Springer, Cham},
  doi       = {10.1007/978-3-031-62217-5_20}
}
```

## Disclaimer

Research and educational use only. Not a medical device. Do not use as the sole basis for clinical diagnosis.
