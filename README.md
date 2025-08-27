# Heart Disease Prediction using Machine Learning

A machine learning solution for predicting heart disease using the UCI Heart Disease dataset with Random Forest classifier.
Develop a Disease Prediction Toolkit by training machine learning models (e.g., logistic
regression, random forest) to predict diseases using real-world health datasets. This
project guides beginners to analyze data, build accurate models, and document results
in a professional portfolio for healthcare AI/ML roles.

## Overview

This project implements a binary classification model to predict the presence of heart disease in patients based on clinical parameters. The solution includes data preprocessing, model training, and a prediction pipeline for real-world deployment.

## Features

- 🏥 **Random Forest Classification** with 85%+ accuracy
- 📊 **Feature Importance Analysis** to identify key predictors
- 🔧 **Complete Data Preprocessing** pipeline
- 💾 **Model Persistence** for production deployment
- 📝 **User Template** for easy predictions

## Dataset

Uses the **UCI Heart Disease Dataset** with clinical features including:

**Numerical**: age, blood pressure, cholesterol, max heart rate, etc.
**Categorical**: gender, chest pain type, ECG results, etc.
**Target**: Binary classification (0 = No Disease, 1 = Disease)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib joblib
```

## Usage

### 1. Train the Model
```bash
python heart_disease_prediction.py
```

This generates:
- `heart_disease_prediction.pkl` (trained model)
- `heart_scaler.pkl` (feature scaler)
- `heart_user_template.csv` (input template)

### 2. Make Predictions

**Using CSV file:**
```python
from heart_disease_prediction import predict_user_input

# Predict using your data
results = predict_user_input("patient_data.csv", "predictions.csv")
```

**Programmatic prediction:**
```python
import joblib

# Load saved model
model = joblib.load("heart_disease_prediction.pkl")
scaler = joblib.load("heart_scaler.pkl")

# Make prediction (after preprocessing)
prediction = model.predict(scaled_data)
```

## Model Performance

```
Random Forest Accuracy: 85.25%

Classification Report:
              precision  recall  f1-score  support
           0      0.87     0.85     0.86       33
           1      0.84     0.86     0.85       28
```

## Project Structure

```
heart-disease-prediction/
├── heart_disease_prediction.py    # Main script
├── heart_disease_uci.csv         # Dataset
├── heart_user_template.csv       # Input template
├── *.pkl files                   # Saved models
└── README.md
```

## Key Functions

### `predict_user_input(user_csv, output_csv)`
Makes predictions on user data and saves results.

**Parameters:**
- `user_csv`: Path to patient data CSV
- `output_csv`: Output file path

## Data Preprocessing

- **Missing Values**: Numerical → mean, Categorical → "Unknown"
- **Encoding**: One-hot encoding for categorical features  
- **Scaling**: StandardScaler for numerical features
- **Target**: Convert to binary (0/1) classification

## Technical Details

- **Algorithm**: Random Forest (100 estimators)
- **Train/Test Split**: 80/20
- **Features**: Handles mixed numerical/categorical data
- **Output**: Binary predictions with probability scores

## Medical Disclaimer

⚠️ **For educational purposes only. Not for medical diagnosis. Consult healthcare professionals for medical decisions.**

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Create Pull Request

## License

MIT License - see LICENSE file for details.

## Contact

- **GitHub**: [GitHub](https://github.com/altamash8986)
- **Email**: mohdaltamash37986@gmail.com

---
*Keywords: machine learning, heart disease prediction, random forest, healthcare AI, python*
