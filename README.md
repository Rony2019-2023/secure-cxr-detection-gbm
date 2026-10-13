COVID-19 Detection from Chest X-Ray Images: GBM and 10 Other Models (Original vs. Updated on 7,001 Images)
Comparative evaluation of 11 machine learning models for binary COVID-19 detection from chest X-ray (CXR) images on an expanded dataset of 7,001 images (COVID vs non-COVID, where non-COVID = Normal + Viral pneumonia), reporting sensitivity, specificity, F1, ROC-AUC and PR-AUC instead of accuracy alone. This repository contrasts the original published findings on 2,905 images with an expanded, robust experiment, addressing previous limitations such as severe class imbalance, tiny test sets, and low COVID recall, with a long-term vision of building Trustworthy and Robust Medical AI systems.

Publication
COVID-19 Detection from Chest X-Ray Images Using GBM with Comparative Analysis
Abisek Dahal, Abu Motaleb Rony, Soumen Moulik
Springer CCIS 2128 (MIND 2023, published 2024), Scopus indexed
DOI: 10.1007/978-3-031-62217-5_20

Original paper: GBM and CNN reached 97.41% accuracy; GBM needed less computation time than CNN.

This repository: An updated pipeline on 7,001 CXR images with binary labels. The focus is COVID sensitivity and balanced metrics, not headline accuracy alone.

The paper's numbers and these numbers are not directly comparable. The datasets, class balance and test sets differ (see the comparison below).

1. Task Setup & Implementation Details
Item	Detail
Task	Binary classification
COVID (1)	COVID-19 positive CXR
Non-COVID (0)	Normal + Viral pneumonia, merged into one class
Dataset	COVID-19 Radiography Database (Kaggle), three class folders in a FullData directory (add the download date or version)
Images	7,001: COVID-19 3,618, Normal 2,038, Viral pneumonia 1,345 (non-COVID total 3,383)
Split	80% train / 20% test, stratified, random_state=42 (5,600 / 1,401)
Test set	724 COVID-19, 677 non-COVID (399 Normal, 278 Viral pneumonia)
Input	224 x 224, grayscale, scaled to [0, 1]
Image selection: All COVID-19 and Viral pneumonia images were used. For the Normal class, the first 2,038 images by file number were taken (about Normal-1 to Normal-2038). This is not a random sample and its effect was not tested.

Labeling rule: Every Normal and Viral pneumonia image is labeled 0, every COVID-19 image is labeled 1. This matches a practical screening goal: COVID vs everything else.

2. Limitations Removed & What Changed
Limitation in the Published Paper	Status in This Updated Pipeline
Very small test set (reported recalls were multiples of 1/8, consistent with ~8 COVID test images)	Fixed: Expanded test set to 1,401 images (724 COVID-19, 677 non-COVID) with confidence intervals.
Severe class imbalance (strongly imbalanced dataset)	Addressed: Expanded to 7,001 images, resulting in a nearly balanced dataset (~51.7% COVID).
Very low COVID recall (0.25 to 0.63 for classical models)	Improved: Recall increased to 0.95–0.98 for top models, significantly reducing missed COVID cases.
Misleading high accuracy on a tiny, imbalanced test set	Addressed: Reported F1, Specificity, ROC-AUC, and PR-AUC alongside accuracy.
Sensitivity formula error in Eq. (6) (Sen = TP / (TP+FP))	Corrected: Used the true sensitivity/recall formula (Sen = TP / (TP+FN)).
3. Original Paper Results (2,905 Images Baseline)
Original Evaluation Metrics (Table 1 from Paper)

Algorithm	Precision	Recall (Sensitivity)	F1-Score	Specificity	Accuracy
CNN	1.000	0.375	0.545	1.000	0.974
SVM	0.500	0.625	0.555	0.972	0.958
K-NN	0.800	0.500	0.615	0.994	0.974
Decision Tree	0.166	0.375	0.230	0.918	0.896
Naive Bayes	0.161	0.625	0.256	0.859	0.849
Logistic Regression	0.444	0.500	0.470	0.972	0.953
Random Forest	1.000	0.250	0.400	1.000	0.968
RNN	0.980	0.378	0.600	1.000	0.958
AdaBoost (AdaBoot)	0.500	0.250	0.333	0.989	0.958
XGBoost	1.000	0.250	0.400	1.000	0.968
GBM	1.000	0.370	0.540	1.000	0.974
Original Computation Time (Table 2 from Paper)

Algorithm	Training Time (s)	Testing Time (s)	Total Time (s)
CNN	980.25	60.23	1040.48
SVM	870.49	20.95	891.44
K-NN	70.85	12.18	83.03
Decision Tree	33.12	12.32	45.44
Naive Bayes	71.16	13.25	84.41
Logistic Regression	55.40	12.31	67.71
Random Forest	43.11	12.20	55.31
RNN	204.94	18.78	223.72
AdaBoost	16.32	7.25	23.57
XGBoost	32.89	8.25	41.14
GBM	420.12	32.12	452.24
4. Updated Results (7,001 Images, 1,401 Test Images)
Model	Accuracy	Precision	Recall (Sensitivity)	Specificity	F1	ROC-AUC
CNN	0.964	0.952	0.981	0.947	0.966	0.991
XGBoost	0.936	0.927	0.950	0.920	0.939	0.984
GBM	0.934	0.923	0.950	0.916	0.937	0.983
SVM	0.931	0.899	0.975	0.883	0.936	0.979
KNN	0.928	0.901	0.967	0.886	0.933	0.968
Random Forest	0.926	0.906	0.957	0.894	0.931	0.980
Logistic Regression	0.882	0.877	0.896	0.866	0.887	0.942
Decision Tree	0.865	0.860	0.883	0.846	0.871	0.839
AdaBoost	0.847	0.836	0.876	0.817	0.856	0.923
Naive Bayes	0.841	0.813	0.899	0.778	0.854	0.844
RNN	0.742	0.776	0.703	0.783	0.738	0.803
5. Future Work: Towards a Trustworthy & Robust Medical AI
Building on this foundational research, our ultimate goal is to transition from standard classification accuracy to deploying Trustworthy, Secure, and Clinically Viable Medical AI. The upcoming roadmap includes:

Adversarial Robustness & Cybersecurity for AI: Implementing defense mechanisms (using tools like Adversarial Robustness Toolbox - ART) to secure models against adversarial attacks and malicious manipulations on medical imagery.
Federated Learning for Healthcare: Developing privacy-preserving decentralized learning frameworks (using frameworks like Flower) so multiple hospitals can collaboratively train robust diagnostic models without sharing sensitive patient data.
Explainable AI (XAI): Integrating visual interpretability techniques such as Grad-CAM and SHAP heatmaps to highlight exact regions in chest X-rays that drive model decisions, fostering trust with clinicians.
Rigorous Validation & Cross-Validation: Moving beyond a single split to Stratified k-fold cross-validation, performing patient-level data splitting to prevent data leakage, and conducting multi-center/external validation.
Advanced Feature Extraction & Transfer Learning: Incorporating pre-trained deep transfer-learning architectures (ResNet, EfficientNet) and feature dimensionality reduction (PCA) for classical models instead of raw high-dimensional pixels.
Authors
Abisek Dahal, NIT Meghalaya
Abu Motaleb Rony, NIT Meghalaya (B.Tech CSE); Instructor, NIST Nilphamari
Email: abumotalebrony@gmail.com
GitHub: motaleb-roni, Rony2019-2023
Soumen Moulik, NIT Meghalaya
Citation
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
Disclaimer
Research and educational use only. Not a medical device. Do not use as the sole basis for clinical diagnosis. Do not use as the sole basis for clinical diagnosis.
