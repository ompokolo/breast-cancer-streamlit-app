import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Breast Cancer Diagnostic Classifier")
st.write("Upload a breast cancer dataset to train and test the model.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)

    # Assign column names
    column_names = [
        "ID", "Diagnosis", "Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean", "Smoothness_Mean",
        "Compactness_Mean", "Concavity_Mean", "Concave_Points_Mean", "Symmetry_Mean", "Fractal_Dimension_Mean",
        "Radius_Se", "Texture_Se", "Perimeter_Se", "Area_Se", "Smoothness_Se", "Compactness_Se", "Concavity_Se",
        "Concave_Points_Se", "Symmetry_Se", "Fractal_Dimension_Se", "Radius_Worst", "Texture_Worst", "Perimeter_Worst",
        "Area_Worst", "Smoothness_Worst", "Compactness_Worst", "Concavity_Worst", "Concave_Points_Worst", "Symmetry_Worst",
        "Fractal_Dimension_Worst"
    ]
    df.columns = column_names

    # Data Preprocessing
    df.drop(columns=["ID"], inplace=True)
    df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

    # Splitting features and target
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    st.success(f"Model trained! Accuracy on test set: {accuracy:.2%}")

    st.subheader("Make a Prediction")
    user_input = []
    for feature in X.columns:
        val = st.number_input(f"{feature}", min_value=0.0, format="%f")
        user_input.append(val)

    if st.button("Predict"):
        scaled_input = scaler.transform([user_input])
        prediction = model.predict(scaled_input)[0]
        result = "Malignant" if prediction == 1 else "Benign"
        st.info(f"Prediction: {result}")
