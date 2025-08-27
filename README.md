# Heart Disease Prediction System 🫀

[![Python](https://img.shields.io/badge/python-v3.10-blue?logo=python)](https://www.python.org/)  
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.2-green?logo=scikit-learn)](https://scikit-learn.org/stable/)  
[![Pandas](https://img.shields.io/badge/pandas-v1.6-blue?logo=pandas)](https://pandas.pydata.org/)  
[![Matplotlib](https://img.shields.io/badge/matplotlib-v3.7-orange?logo=matplotlib)](https://matplotlib.org/)  

A **machine learning-based system** to predict the likelihood of **heart disease** in individuals using the **UCI Heart Disease dataset**.  
This project implements a **Random Forest Classifier** with **data preprocessing, feature scaling, one-hot encoding, and model persistence**, providing users with a **ready-to-use CSV template for predictions**.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Data Preprocessing](#data-preprocessing)  
- [Model Details](#model-details)  
- [Predictions](#predictions)  
- [Feature Importance & Visualization](#feature-importance--visualization)  
- [Future Enhancements](#future-enhancements)  
- [License](#license)  
- [Author](#author)  

---

## Overview 🩺

Heart disease is one of the leading causes of death globally. Predicting heart disease early can **save lives** and help doctors provide timely treatment.  
This project uses a **Random Forest Machine Learning model** to classify whether a person is at risk of heart disease based on **clinical attributes**.  

The system is **user-friendly**: users can input a CSV file of patient data and receive predictions instantly, along with a **sample CSV template** to guide input.

---

## Features ✨

- Binary classification:  
  - `0` = No Heart Disease  
  - `1` = Heart Disease  
- Automatically handles **missing values** in numeric and categorical columns  
- **One-hot encoding** for categorical variables  
- **StandardScaler** for feature scaling  
- Visualizes **top 10 important features**  
- Saves **trained model** and **scaler** for future predictions  
- Provides **user template CSV** for easy data input  
- Outputs **prediction CSV** for multiple patients at once  

---

## Dataset 📊

**Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)  

**Features include:**  
- Numeric: `age`, `trestbps` (resting blood pressure), `chol` (cholesterol), `thalach` (maximum heart rate), etc.  
- Categorical: `sex`, `cp` (chest pain type), `fbs` (fasting blood sugar), `restecg` (resting ECG results), `exang` (exercise-induced angina), `slope`, `thal`  
- Target: `num` (0 or 1 for absence/presence of heart disease)  

The dataset has been preprocessed to **handle missing values, normalize numeric features, and encode categorical columns**.

---

## Installation ⚙️

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
