# Titanic Survival Prediction: Comparative ML Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📄 Project Overview
This project focuses on predicting the survival of passengers on the Titanic using various Machine Learning algorithms. The goal was to analyze the dataset, preprocess raw data, handle missing values, and compare the performance of multiple classification models to identify the most accurate predictor.

This repository serves as a demonstration of:
* Data Cleaning & Preprocessing (Pandas/NumPy)
* Exploratory Data Analysis
* Model Training & Evaluation (Scikit-Learn)

## 📂 Dataset
The dataset used is the standard Titanic dataset (Kaggle), containing passenger details such as:
* **Features:** Age, Sex, Embarked location, Passenger Class.
* **Target Variable:** Survived (0 = No, 1 = Yes).

## 🛠️ Methodology

### 1. Data Preprocessing
* **Feature Selection:** Removed non-essential columns (`PassengerId`, `Name`, `Ticket`, `Cabin`) to reduce noise.
* **Missing Value Imputation:**
    * *Age:* Imputed using a statistical approach (random values generated within one standard deviation of the mean) to maintain distribution.
    * *Embarked:* Filled missing values with the mode ('S').
* **Encoding:** Converted categorical variables (`Sex`, `Embarked`) into numeric formats for machine learning compatibility.
* **Scaling:** Applied `StandardScaler` to normalize feature distribution, ensuring models like KNN and SVM perform optimally.

### 2. Models Implemented
The following classifiers were trained and tested:
1.  **Logistic Regression** (Baseline model)
2.  **Support Vector Classifier (SVC)**
3.  **Decision Tree Classifier**
4.  **Random Forest Classifier** (Ensemble method)
5.  **K-Nearest Neighbors (KNN)**
6.  **Gaussian Naive Bayes**

## 🚀 How to Run

### Prerequisites
Ensure you have Python installed along with the required libraries:

```bash
pip install numpy pandas scikit-learn
