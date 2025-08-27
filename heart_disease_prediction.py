import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv("heart_disease_uci.csv")

# Handle missing values
numeric_cols = dataset.select_dtypes(include="number").columns.tolist()
dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())

# Handle categorical columns
cat_cols = dataset.select_dtypes(include="object").columns.tolist()
if "num" in cat_cols:  # Ensure target column isn't included
    cat_cols.remove("num")

# Features & Target
X = dataset.drop("num", axis=1)
y = (dataset["num"] > 0).astype(
    int
)  # Binary classification (0 = No Disease, 1 = Disease)

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=cat_cols)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred_rf = rf_model.predict(X_test_scaled)
print(f"Random Forest Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Feature Importance Plot
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind="barh", figsize=(8, 5))
plt.title("Top 10 Important Features (Random Forest)")
plt.show()

# Save model and scaler
joblib.dump(rf_model, "heart_disease_prediction.pkl")
joblib.dump(scaler, "heart_scaler.pkl")

# Save sample template for user input
sample = X.head(1)
sample.to_csv("heart_user_template.csv", index=False)
print("User Template saved as heart_user_template.csv")


def predict_user_input(user_csv, output_csv="user_prediction_output.csv"):

    # Load user input
    user_data = pd.read_csv(user_csv)

    # One-hot encode categorical variables
    user_data_encoded = pd.get_dummies(user_data)

    # Reindex to match training columns (handles missing & extra columns)
    user_data_encoded = user_data_encoded.reindex(columns=X.columns, fill_value=0)

    # Load scaler and model
    scaler = joblib.load("heart_scaler.pkl")
    model = joblib.load("heart_disease_prediction.pkl")

    # Scale input
    user_scaled = scaler.transform(user_data_encoded)

    # Predictions
    predictions = model.predict(user_scaled)

    # Append predictions to user data
    user_data["Heart_Disease_Prediction"] = predictions

    # Save results
    user_data.to_csv(output_csv, index=False)

    # ✅ Print EXACT output format
    print(user_data[["Heart_Disease_Prediction"]].to_string(index=True))

    return user_data


predict_user_input("heart_user_template.csv")


user_df = pd.read_csv("heart_disease_uci.csv")

# GETTING COLUMNS LIST FROM TRAINING DATAFRAME
numeric_cols = dataset.select_dtypes(include="number").columns.tolist()
cat_cols = dataset.select_dtypes(include="object").columns.tolist()
bools_cols = dataset.select_dtypes(include="bool").columns.tolist()


# DROPPING COLUMNS WHICH ARE EXTRA IN USER_DF THAN REQUIRED TP
numeric_cols = [col for col in numeric_cols if col in user_df.columns]
cat_cols = [col for col in cat_cols if col in user_df.columns]
bool_cols = [col for col in bools_cols if col in user_df.columns]


# FILL THE MISSING NUMERIC COLUMNS , CAT COLUMN
user_df[numeric_cols] = user_df[numeric_cols].fillna(user_df[numeric_cols].mean())

for col in cat_cols:
    user_df[col] = user_df[col].fillna("Unknown")

    for col in bool_cols:
        user_df[col] = user_df[col].astype(int)


# ONE HOT ENCODING CAT COLUMNS
user_df_encoded = pd.get_dummies(user_df, columns=cat_cols)


# ALIGN COLUMNS
user_df_encoded = user_df_encoded.reindex(columns=X.columns, fill_value=0)

# SCALER DATA
scaler = joblib.load("heart_scaler.pkl")
user_df_scaled = scaler.transform(user_df_encoded)


# PREDICTION
rf_model = joblib.load("heart_disease_prediction.pkl")
pred = rf_model.predict(user_df_scaled)
user_df["heart_disease_prediction"] = pred

# SHOW RESULT
print(user_df)

