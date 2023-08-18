# Breast Cancer Diagnosis and Survival Prediction

Welcome to the "Breast Cancer Diagnosis and Survival Prediction" repository. This project is designed to leverage machine learning techniques for predicting breast cancer diagnosis and post-surgery survival. The provided R code snippets encompass a comprehensive workflow, including data preprocessing, model training, hyperparameter tuning, prediction, and result visualization.

## Table of Contents

1. [Introduction](#introduction)
2. [Code Description](#code-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Model Comparison](#model-comparison)
6. [Final Model Tuning and Prediction](#final-model-tuning-and-prediction)
7. [Prediction Sample](#integration)
8. [Results](#results)
9. [Usage](#usage)
10. [Requirements](#requirements)
11. [Dependencies](#dependencies)
12. [Contributing](#contributing)
13. [Acknowledgments](#acknowledgments)
14. [References](#references)
15. [License](#license)

## 1. Introduction

Our goal is to utilize machine learning models to predict breast cancer diagnosis and patient survival. This project encompasses two distinct datasets, each serving a specific purpose:

- **Dataset 1: Breast Cancer Diagnosis using SVM**

   This dataset focuses on breast cancer classification, employing Support Vector Machines (SVMs) to develop a robust predictive model. Patient attributes such as age, gender, biomarker levels, tumor stage, histology, and receptor statuses (ER, PR, HER2) are considered. By preprocessing the data, selecting relevant features, building and tuning the SVM model, and conducting comprehensive evaluations, we achieve an accurate classifier that aids in distinguishing malignant from benign breast tumors.

- **Dataset 2: Breast Cancer Survival Prediction using Patient Data**

   Centered on predicting post-surgery survival, this dataset employs machine learning techniques, including the Random Survival Forest. Patient characteristics such as age, gender, biomarkers, tumor stage, histology, and receptor status are utilized. The resulting survival prediction model stratifies patients based on predicted survival probabilities, facilitating tailored treatment planning and interventions.

## 2. Code Description

### Data Preprocessing

1. Data preprocessing: Load and preprocess the breast cancer dataset, which includes handling missing values and duplicates in the dataset.
2. Handling Outliers and Visualization: Summary statistics of the data and handling outliers by replacing them with upper and lower bounds found through IQR
3. Data balancing: Use SMOTE (Synthetic Minority Over-Sampling Technique) to balance the class distribution of the target variable.
4. Standard Normalization of Data: Get all the data points to one measurable scale so that the model doesn't get influenced by high variance.

Overall, our code initiates with data preprocessing steps, and standardizing numeric features. Then we synthesize the data to balance the data and not get influenced by Benign cases, and the data is prepared for further analysis.

![Data Preprocessing](https://github.com/DeekshaLokesh331/DeekshaLokesh/blob/main/image1.png)
Figure 1:Balanced Dataset 

![Data Preprocessing](https://github.com/DeekshaLokesh331/DeekshaLokesh/blob/main/image4.png)
Figure 2: Types of Surgeries Performed on Patients

### 3. Model Training and Evaluation

The following models are trained and evaluated using k-fold cross-validation:

- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree

For both diagnosis and survival prediction, optimal hyperparameters are determined using predefined grids (`tuneGrid` and `tuneGrid1`). F1 scores are computed for each fold, with overall mean F1 scores showcased for each model.

### 4. Model Comparison

To aid comparison, F1 scores among various models (Random Forest, SVM, Logistic Regression, and Decision Tree) are visualized through a side-by-side bar plot.

![Model Comparison](https://github.com/DeekshaLokesh331/DeekshaLokesh/blob/main/image3.png)
Figure 3: F1 score comparison for all the models 

### 5. Final Model Tuning and Prediction

Following hyperparameter identification, final models are trained and tested using these parameters. Predictions are then generated on test data, accompanied by actual vs. predicted counts and a confusion matrix for model assessment.

![Final Model Tuning and Prediction](https://github.com/DeekshaLokesh331/DeekshaLokesh/blob/main/image2.png)
Figure 4: Model metric for Diagnosis after tuning
  
### 6. Prediction Sample

Our code provides diagnosis and survival predictions (`pred1` and `pred2`). Diagnosis prediction prompts further analysis or surgery recommendations, while survival prediction estimates "Alive" or "Dead" status post-surgery.
![Prediction Sample](https://github.com/DeekshaLokesh331/DeekshaLokesh/blob/main/image5.jpg)
Figure 5: Sample prediction based on the provided input data

## 8. Results

This project yields the following outcomes:

- Model performance metrics, including F1 scores for diverse machine learning algorithms.
- Visualizations comparing F1 scores of different models.
- Predictions concerning cancer diagnosis and survival on test data.

## 9. Usage

To utilize this code, follow these steps:

1. Ensure R is installed on your machine.
2. Clone this repository to your local environment.
3. Download and Install the `DMwr` package using this URL: [DMwR Package](https://cran.r-project.org/src/contrib/Archive/DMwR/DMwR_0.4.1.tar.gz) for data synthesis.
4. Modify data paths (`breast-cancer.csv` and `BRCV.csv`) in your code if necessary to load the data.
5. Execute the R script containing the code.

## 10. Requirements

- R programming language
- Fundamental understanding of machine learning and R programming

## 11. Dependencies

The R code in this project relies on various R libraries, including but not limited to:

- `DMwR`: For data synthesis
- `randomForest`: For constructing random forest models
- `glm`: For logistic regression models
- `e1071`: For support vector machine models
- `party`: For decision tree models
- `ggplot2`: For data visualization

## 12. Contributing

Contributions to this project are encouraged. Feel free to fork the repository and submit pull requests.

## 13. Acknowledgments

This project is developed for educational purposes and is not intended to replace medical advice. Model accuracy relies on data quality and selected features. Always consult medical professionals for precise diagnosis and treatment decisions. Your contributions to this project further advance our understanding of breast cancer prediction and survival.

## 14. References

### Breast Cancer Diagnosis:

1. S. Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," Nature, 2017. (While this paper focuses on skin cancer, it's a landmark example of using deep learning for medical image classification and can be analogous to breast cancer diagnosis.)

2. L. Hou et al., "Patch-based Convolutional Neural Network for Whole Slide Tissue Image Classification," in IEEE Journal of Biomedical and Health Informatics, 2020.

### Breast Cancer Survival Prediction:

1. M. Fornaciari et al., "Machine Learning for Predicting the Outcome of Patients with Breast Cancer: A Review," Frontiers in Oncology, 2019.

2. S. Li et al., "Predicting breast cancer survival using deep learning techniques," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2019.

3. Y. Liu et al., "A novel machine learning approach to predict postoperative survival of cancer patients," Sci Rep, 2017.

## 15. License

The MIT License (MIT) 2023 - Parinitha Kiran and Deeksha Lokesh.
