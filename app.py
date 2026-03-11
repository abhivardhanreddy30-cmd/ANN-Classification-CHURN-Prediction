import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Load encoders and scaler
with open("lable_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder.pkl", "rb") as file:
    onehot_encoder = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Title
st.title("Customer Churn Prediction")

# User Inputs
country = st.selectbox("Country", onehot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)

credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 40)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 200000.0, 50000.0)
num_of_products = st.number_input("Number of Products", 1, 4, 1)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# Encode Gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# Create DataFrame
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_credit_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})


# Encode country
country_df = pd.DataFrame(
    [[country]], columns=onehot_encoder.feature_names_in_)
country_encoded = onehot_encoder.transform(country_df).toarray()

country_encoded_df = pd.DataFrame(
    country_encoded,
    columns=onehot_encoder.get_feature_names_out()
)

# Merge
input_data = pd.concat(
    [input_data.reset_index(drop=True), country_encoded_df], axis=1)

# Ensure all training columns exist
for col in scaler.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

# Correct column order
input_data = input_data[scaler.feature_names_in_]

# Scale
input_data_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)

    if prediction[0][0] > 0.5:
        st.error("Customer is likely to CHURN ❌")
    else:
        st.success("Customer is NOT likely to churn ✅")

    st.write("Churn Probability:", float(prediction[0][0]))