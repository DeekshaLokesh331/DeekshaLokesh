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
14. [License](#license)

## 1. Introduction

Our goal is to utilize machine learning models to predict breast cancer diagnosis and patient survival. This project encompasses two distinct datasets, each serving a specific purpose:

- **Dataset 1: Breast Cancer Diagnosis using SVM**

   This dataset focuses on breast cancer classification, employing Support Vector Machines (SVMs) to develop a robust predictive model. Patient attributes such as age, gender, biomarker levels, tumor stage, histology, and receptor statuses (ER, PR, HER2) are considered. By preprocessing the data, selecting relevant features, building and tuning the SVM model, and conducting comprehensive evaluations, we achieve an accurate classifier that aids in distinguishing malignant from benign breast tumors.

- **Dataset 2: Breast Cancer Survival Prediction using Patient Data**

   Centered on predicting post-surgery survival, this dataset employs machine learning techniques, including the Random Survival Forest. Patient characteristics such as age, gender, biomarkers, tumor stage, histology, and receptor statuses are utilized. The resulting survival prediction model stratifies patients based on predicted survival probabilities, facilitating tailored treatment planning and interventions.

## 2. Code Description

### Data Preprocessing

Our code initiates with data preprocessing steps, standardizing numeric features and transforming categorical features into binary indicators using the `model1_par` and `model2_par` functions. After synthesizing data to balance the data and not get influenced by benign cases, the data is prepared for further analysis.

![Data Preprocessing](/Breast%20cancer%202/Final%20BC/image2.png)
Figure 1:Balance Dataset 

### 3. Model Training and Evaluation

The following models are trained and evaluated using k-fold cross-validation:

- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree

For both diagnosis and survival prediction, optimal hyperparameters are determined using predefined grids (`tuneGrid` and `tuneGrid1`). F1 scores are computed for each fold, with overall mean F1 scores showcased for each model.

### 4. Model Comparison

To aid comparison, F1 scores among various models (Random Forest, SVM, Logistic Regression, and Decision Tree) are visualized through a side-by-side bar plot.

![Model Comparison](/Breast%20cancer%202/Final%20BC/image2.png)
Figure 2: F1 score comparison for all the models 

### 5. Final Model Tuning and Prediction

Following hyperparameter identification, final models are trained and tested using these parameters. Predictions are then generated on test data, accompanied by actual vs. predicted counts and a confusion matrix for model assessment.

![Final Model Tuning and Prediction](/Breast%20cancer%202/Final%20BC/image3.png)
Figure 3: Model metric for Diagnosis after tuning
  
### 6. Prediction Sample

Our code provides diagnosis and survival predictions (`pred1` and `pred2`). Diagnosis prediction prompts further analysis or surgery recommendations, while survival prediction estimates "Alive" or "Dead" status post-surgery.

## 8. Results

This project yields the following outcomes:

- Model performance metrics, including F1 scores for diverse machine learning algorithms.
- Visualizations comparing F1 scores of different models.
- Predictions concerning cancer diagnosis and survival on test data.

## 9. Usage

To utilize this code, follow these steps:

1. Ensure R is installed on your machine.
2. Clone this repository to your local environment.
3. Install the `DMwr` package using this URL: [DMwR Package](https://cran.r-project.org/src/contrib/Archive/DMwR/DMwR_0.4.1.tar.gz) for data synthesis.
4. Modify data paths (`breast-cancer.csv` and `BRCV.csv`) in your code.
5. Verify that the `model1_par` and `model2_par` functions are compatible with your data and domain.
6. Execute the R script containing the code.

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

## 14. License

The MIT License (MIT) 2023 - Deeksha Lokesh and Parinitha Kiran.
