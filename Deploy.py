#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pickle import load
import numpy as np
import time
from PIL import Image

im = Image.open('icon4.png')
st.set_page_config(page_title="Customer Churn Prediction App",page_icon=im)
html_temp = """
    <div style="background-color:#f63350 ;padding:10px">
    <h2 style="color:white;text-align:center;">
    CUSTOMER CHURN PREDICATION APP </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
st.image('title.png')
st.sidebar.title("USER INPUT")

model = load(open('churn_prediction.sav','rb'))

# Manual mapping of encoded values to categories
encoded_to_gender_mapping = {
    "0": "Female",
    "1": "Male"
}

encoded_to_geography_mapping = {
    "0": "France",
    "1": "Germany",
    "2": "Spain"
}

encoded_to_IsActiveMember_mapping = {
    "0": "No",
    "1": "Yes"
}

# Define tooltips
tooltips = {
    "CustomerId": "Unique identifier for the customer",
    "CreditScore": "Credit score of the customer",
    "Geography": "Country where the customer is located",
    "Gender": "Gender of the customer",
    "Age": "Age of the customer",
    "Balance": "Account balance of the customer",
    "IsActiveMember": "Indicates whether the customer is an active member",
    "EstimatedSalary": "Estimated salary of the customer"
}

def classify_churn(CustomerId, CreditScore, Geography, Gender, Age, Balance, IsActiveMember, EstimatedSalary):
    
    Gender = Gender.strip()
    encoded_gender = None
    original_gender = 'Unknown'
    if Gender in encoded_to_gender_mapping.values():
        for key, value in encoded_to_gender_mapping.items():
            if value == Gender:
                encoded_gender = key
                original_gender = Gender
                break

    if encoded_gender is None:
        st.error(f"Invalid gender: {Gender}")
        return None
    
    Geography = Geography.strip()
    encoded_geography = None
    original_geography = 'Unknown'
    if Geography in encoded_to_geography_mapping.values():
        for key, value in encoded_to_geography_mapping.items():
            if value == Geography:
                encoded_geography = key
                original_geography = Geography
                break

    if encoded_geography is None:
        st.error(f"Invalid response: {Geography}")
        return None
    
    IsActiveMember = IsActiveMember.strip()
    encoded_IsActiveMember = None
    original_IsActiveMember = 'Unknown'
    if IsActiveMember in encoded_to_IsActiveMember_mapping.values():
        for key, value in encoded_to_IsActiveMember_mapping.items():
            if value == IsActiveMember:
                encoded_IsActiveMember = key
                original_IsActiveMember = IsActiveMember
                break

    if encoded_IsActiveMember is None:
        st.error(f"Invalid response: {IsActiveMember}")
        return None
    
    input_features = np.array([[CustomerId, CreditScore, encoded_geography, encoded_gender, Age, Balance, encoded_IsActiveMember,                                           EstimatedSalary]] ,dtype=np.float64)

    # Print the input features 
    st.write("Input features:")
    st.write(pd.DataFrame(input_features, columns=['CustomerId', 'CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'IsActiveMember',                                                      'EstimatedSalary']))
    
    try:
        # Show loading message
        with st.spinner('Predicting...'):
            #Predict using the model
            predicted_class = model.predict(input_features)
            predicted_probability = model.predict_proba(input_features)[0][1]  # Probability of positive class
            return predicted_class[0], predicted_probability
    
    except Exception as e:
        # Handle any exceptions that occur during prediction
        st.error(f"An error occurred during prediction: {e}")
        return None
    
def main():
    CustomerId = st.sidebar.text_input("Customer Id", help=tooltips["CustomerId"])
    CreditScore = st.sidebar.text_input("Credit Score", help=tooltips["CreditScore"])
    Geography = st.sidebar.selectbox("Geography", ["France", "Germany","Spain"], help=tooltips["Geography"])
    Gender = st.sidebar.selectbox("Gender", ["Female", "Male"], help=tooltips["Gender"])
    Age = st.sidebar.text_input("Age", help=tooltips["Age"])
    Balance = st.sidebar.number_input("Balance", help=tooltips["Balance"])
    IsActiveMember = st.sidebar.selectbox("Are you an active member?", ["No", "Yes"], help=tooltips["IsActiveMember"])
    EstimatedSalary = st.sidebar.number_input("Estimated Salary", help=tooltips["EstimatedSalary"])
        
    if st.sidebar.button("Predict"):
        if not CustomerId or not CreditScore or not Age:
            st.error("Please provide values for Customer Id, Credit Score, and Age.")
        else:
            try:
                CustomerId = float(CustomerId)
                CreditScore = float(CreditScore)
                Age = float(Age)
                
                predicted_class, predicted_probability = classify_churn(CustomerId, CreditScore, Geography, Gender, Age, Balance,                                                                                   IsActiveMember, EstimatedSalary)
                if predicted_class is not None and predicted_probability is not None:
                    if predicted_class == 1:
                        st.success('Customer likely to churn😒📉')
                        st.info(f"Predicted Probability: {predicted_probability:.2f}")
                        time.sleep(.5)
                        st.empty() 

                    else:
                        st.success('Customer unlikely to churn😄📈')
                        st.info(f"Predicted Probability: {1 - predicted_probability:.2f}")
                        time.sleep(.5)
                        st.empty()
                        
            except ValueError:
                st.error("Invalid input format for Customer Id, Credit Score, or Age. Please provide numeric values.")
                
if __name__ == '__main__':
    main()
                   



