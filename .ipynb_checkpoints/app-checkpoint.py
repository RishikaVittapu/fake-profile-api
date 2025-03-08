import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature names
model = joblib.load("fake_profile_detector.pkl")
features = joblib.load("model_features.pkl")

# Streamlit UI
st.title("Fake Profile Detector")
st.write("Enter profile details to predict whether it is fake or real.")

# Create input fields dynamically based on features
user_input = {}
for feature in features:
    user_input[feature] = st.text_input(f"Enter {feature}", "0")

# Convert input to DataFrame
if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=features)
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert types
    
    # Predict
    prediction = model.predict(input_df)[0]
    result = "Fake Profile" if prediction == 1 else "Real Profile"
    
    # Display result
    st.success(f"Prediction: {result}")
