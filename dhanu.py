import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load your trained model
model = tf.keras.models.load_model("demand prediction.h5")

# Load the scaler from the scaler.pkl file
scaler = joblib.load("scaler.pkl")

# Title and description
st.title("Demand Prediction using LSTM Model")
st.write("This app predicts demand based on input features using a trained LSTM model.")

# Input fields for features
year = st.number_input("Year", min_value=2000, max_value=2100, value=2023)
location = st.number_input("Location ID", min_value=0, max_value=100, value=1)
week = st.number_input("Week", min_value=1, max_value=53, value=1)

# Prediction button
if st.button("Predict Demand"):
    # Prepare input data for prediction
    input_data = np.array([[year, location, week]])
    
    # Use the loaded scaler to transform the input data
    input_scaled = scaler.transform(input_data)
    
    # Reshape the input data for LSTM model
    input_lstm = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

    # Predict using the loaded model
    prediction = model.predict(input_lstm)
    
    # Display the prediction
    st.write(f"Predicted Demand: {prediction[0][0]:.2f}")

# Display additional information
st.write("Ensure your input values match the expected format and ranges used during model training.")
