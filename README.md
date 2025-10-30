# Heart Disease Prediction Project

## Overview

This project uses machine learning techniques to predict the likelihood of a patient having heart disease based on various clinical features. The analysis involves data exploration, preprocessing, model training, hyperparameter tuning, and evaluation, ultimately identifying XGBoost as the best-performing model for this dataset.



---
## Dataset

The dataset used is `data/heart-disease-dataset.csv`. It contains anonymized patient data with features such as:

* `age`: Age in years
* `sex`: (1 = male; 0 = female)
* `chest_pain_type`: Chest pain type (values 1-4)
* `resting_bp_s`: Resting blood pressure (in mm Hg)
* `cholesterol`: Serum cholesterol in mg/dl
* `fasting_blood_sugar`: (Fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
* `resting_ecg`: Resting electrocardiographic results (values 0, 1, 2)
* `max_heart_rate`: Maximum heart rate achieved
* `exercise_angina`: Exercise induced angina (1 = yes; 0 = no)
* `oldpeak`: ST depression induced by exercise relative to rest
* `st_slope`: The slope of the peak exercise ST segment (values 0-3)
* `target`: Heart disease presence (1 = yes, 0 = no) - **This is the target variable.**

---

## Analysis Workflow

The main analysis is documented in `notebooks/file.ipynb`. Key steps include:

1.  **Loading Data & Initial Exploration (EDA):** Importing libraries, loading the dataset, checking data types, looking for missing values/duplicates, getting summary statistics, and visualizing the target variable distribution.
2.  **Visualization & Correlation:** Creating a correlation heatmap and plotting distributions & boxplots for numerical features to understand relationships and identify outliers/skewness.
3.  **Data Preprocessing:**
    * Handling potential outliers using the IQR capping method.
    * Applying Box-Cox transformation to reduce skewness in the `oldpeak` feature.
    * Standardizing continuous features using `StandardScaler`.
    * Encoding categorical features using One-Hot Encoding (`pd.get_dummies`).
4.  **Feature Importance:**
    * Using an `ExtraTreesClassifier` to estimate the importance of each feature in predicting the target variable.
    * Visualizing the top 15 most important features.
5.  **Model Training & Baseline Evaluation:**
    * Splitting the preprocessed data into training (80%) and testing (20%) sets, stratified by the target variable.
    * Training a variety of baseline classification models:
        * Logistic Regression
        * Support Vector Machine (SVM)
        * Decision Tree
        * Random Forest
        * Gaussian Naive Bayes
        * K-Nearest Neighbors (KNN)
        * Perceptron
        * Multi-Layer Perceptron (MLP)
        * Gradient Boosting
        * XGBoost
    * Evaluating these models on the test set using Accuracy, F1 Score, and ROC AUC score.
6.  **Hyperparameter Tuning:**
    * Selecting the top-performing baseline models (Random Forest and XGBoost).
    * Using `GridSearchCV` with 5-fold cross-validation to find the optimal hyperparameters for these models based on accuracy.
7.  **Final Model Evaluation:**
    * Evaluating the performance of the tuned RandomForest and XGBoost models on the test set.
8.  **Model Saving:**
    * Saving the best model (tuned XGBoost) to the `models/` directory using `joblib`.

---

## Results Summary

* EDA indicated the dataset was relatively clean (no missing values found). Feature importance highlighted `st_slope`, `max_heart_rate`, and `chest_pain_type` related features as strong predictors.
* Baseline models showed strong performance from tree-based ensembles like Random Forest and XGBoost (~92% accuracy).
* After hyperparameter tuning, the **XGBoost classifier emerged as the best model** with the following approximate scores on the test set:
    * **Accuracy:** 0.9328
    * **F1 Score:** 0.9365
    * **ROC AUC:** 0.9679

---

## How to Set Up and Run

Follow these instructions to run the analysis locally.

**1. Prerequisites:**
   * [Git](https://git-scm.com/) installed
   * [Python 3](https://www.python.org/) (version 3.7+ recommended) installed
   * `pip` (Python package manager) installed

**2. Clone Repository:**
   Open your terminal/command prompt and run:
   ```bash
   git clone <your-repository-url>
   cd heart-disease-project # Or your repository name
