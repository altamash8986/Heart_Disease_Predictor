# Heart Disease Prediction System 🫀

[![Python](https://img.shields.io/badge/python-v3.10-blue?logo=python)](https://www.python.org/)  
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.2-green?logo=scikit-learn)](https://scikit-learn.org/stable/)  
[![Pandas](https://img.shields.io/badge/pandas-v1.6-blue?logo=pandas)](https://pandas.pydata.org/)  

A **machine learning-based system** to predict the likelihood of **heart disease** in individuals using the **UCI Heart Disease dataset**. This project implements a **Random Forest Classifier** and provides a **user-friendly template** to predict disease from custom input CSV files.

---

## Features ✨

- Binary classification: **0 = No Heart Disease, 1 = Heart Disease**  
- Handles **missing numeric and categorical values** automatically  
- Supports **custom user input CSV files** for predictions  
- **One-hot encoding** for categorical variables  
- **Feature scaling** using StandardScaler  
- **Feature importance visualization** for model interpretability  
- Saves **trained model and scaler** for future use  
- User-friendly **CSV template** for predictions  

---

## Dataset 📊

The system uses the **Heart Disease UCI dataset**. Features include:

- Numeric features: `age`, `trestbps`, `chol`, etc.  
- Categorical features: `sex`, `cp`, `fbs`, `restecg`, etc.  
- Target variable: `num` (Heart Disease: 0 or 1)  

Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

---

## Installation ⚙️

1. Clone the repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
