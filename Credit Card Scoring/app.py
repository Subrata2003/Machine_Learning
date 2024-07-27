import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model_filename = 'random_forest_model.pkl'
model = joblib.load(model_filename)

# Define the prediction function
def predict(features):
    df = pd.DataFrame([features])
    prediction = model.predict(df)
    return prediction[0]

# Define the Streamlit app
def main():
    st.title("Credit Scoring Prediction")
    
    # User inputs
    account_balance = st.selectbox("Account Balance", ["no_inf", "little", "moderate", "rich"])
    saving_accounts = st.selectbox("Saving Accounts", ["no_inf", "little", "moderate", "rich"])
    checking_account = st.selectbox("Checking Account", ["no_inf", "little", "moderate", "rich"])
    duration = st.number_input("Duration (in months)", min_value=0)
    credit_amount = st.number_input("Credit Amount", min_value=0)
    age = st.number_input("Age", min_value=0)
    housing = st.selectbox("Housing", ["own", "free", "rent"])
    purpose = st.selectbox("Purpose", ["car", "furniture/equipment", "radio/tv", "education", "business", "domestic appliance", "repairs"])
    sex = st.selectbox("Sex", ["male", "female"])
    
    # Convert inputs to the correct format
    features = {
        'account_balance': account_balance,
        'saving_accounts': saving_accounts,
        'checking_account': checking_account,
        'duration': duration,
        'credit_amount': credit_amount,
        'age': age,
        'housing': housing,
        'purpose': purpose,
        'sex': sex,
    }
    
    if st.button("Predict"):
        prediction = predict(features)
        st.write(f"The predicted credit risk is: {prediction}")

if __name__ == '__main__':
    main()
