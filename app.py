import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Adult Income Classification App")

# Load preprocessing objects
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

# Model dictionary
models = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl",
}

# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))
model = joblib.load(models[model_name])

# File upload
uploaded_file = st.file_uploader("Upload RAW test dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    # Strip whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Encode categorical FEATURE columns only
    for column, encoder in label_encoders.items():
        if column != "income":
            df[column] = df[column].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )

    # Encode target column separately
    df["income"] = df["income"].map({
        "<=50K": 0,
        ">50K": 1
    })

    # Split features and target
    X = df.drop("income", axis=1)
    y = df["income"]

    # Scale features
    X_scaled = scaler.transform(X)

    # Use scaled data for specific models
    if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        predictions = model.predict(X_scaled)
    else:
        predictions = model.predict(X)

    # Metrics
    acc = accuracy_score(y, predictions)

    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {acc:.4f}")

    st.subheader("Evaluation Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy"],
        "Value": [acc]
    })
    st.table(metrics_df)
    st.subheader("Classification Report")
    report = classification_report(y, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, predictions))
