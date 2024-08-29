import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# File paths
model_path = "demand prediction.h5"
scaler_path = "scaler.pkl"

# Load the trained model and scaler
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

try:
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)  # Load the scaler using pickle
except Exception as e:
    st.error(f"Failed to load the scaler: {e}")
    st.stop()

# Title and description
st.title("Demand Prediction using LSTM Model")
st.write("This app predicts demand based on input features using a trained LSTM model.")

# Input fields for features
year = st.number_input("Year", min_value=2000, max_value=2100, value=2023)
location = st.number_input("Location ID", min_value=0, max_value=100, value=1)
week = st.number_input("Week", min_value=1, max_value=53, value=1)
some_other_feature = st.number_input("Other Feature", min_value=0.0, format="%.2f", help="An additional feature.")

# Prediction button
if st.button("Predict Demand"):
    try:
        # Prepare input data for prediction
        input_data = np.array([[year, location, week, some_other_feature]])
        input_scaled = scaler.transform(input_data)  # Use transform instead of fit_transform
        input_lstm = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

        # Predict using the loaded model
        prediction = model.predict(input_lstm)
        
        # Display the prediction
        st.write(f"Predicted Demand: {prediction[0][0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Display additional information
st.write("Ensure your input values match the expected format and ranges used during model training.")
