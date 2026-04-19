import streamlit as st
import pandas as pd
import joblib
import os
import logging
from datetime import datetime

# streamlit run app.py

# הגדרת מערכת הלוגים (תשמור את הנתונים לקובץ app_logs.log)
logging.basicConfig(
    filename='app_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

st.set_page_config(page_title="Loan Checker", page_icon="🏦", layout="centered")

# כותרת האפליקציה
st.title('🏦Loan Approval Checker')

# נתיב הקובץ של המודל המאומן והמעובד
MODEL_PATH = 'loan_model.pkl'

# בדיקה האם המודל קיים
if not os.path.exists(MODEL_PATH):
    st.warning('The system is initializing, please wait')
    st.error('The model file was not found. Please train the model and save it before running the app.')
    logging.warning("App started but model file was not found.")
else:
    st.header('Loan Application Details')

    # טעינת המודל מתוך הקובץ
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    accuracy = model_data['accuracy']

    # יצירת שדות הזנה למשתמש
    applicant_income = st.number_input("Applicant Income (monthly)", min_value=0.0, value=7700.00, step=100.0)
    coapplicant_income = st.number_input(label="Coapplicant Income (monthly)", min_value=0.0, value=1400.00, step=100.0)
    requested_loan_amount = st.number_input(label="Requested Load Amount", min_value=0.0, value=150.00, step=10.0)
    loan_term = st.number_input(label="Loan Term (months)", min_value=0.0, value=360.00, step=12.0)
    credit_history_display = st.selectbox('Credit History', ['Exists (1)', 'Does not exist (0)'])
    credit_history = 1 if credit_history_display == 'Exists (1)' else 0
    marital_status_display = st.selectbox('Marital Status', ['Married', 'Not Married'])
    married = 1 if marital_status_display == 'Married' else 0
    education_display = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    education = 1 if education_display == 'Graduate' else 0

    # כתןר לבדיקה
    if st.button("Check Loan Eligibility"):
        input_data = pd.DataFrame({
            'Married': [married],
            'Education': [education],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [requested_loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history]
        })

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("### 🎉 Congratulations! \n **☑ Loan Approved**")
        else:
            st.error("### 🛑 We're sorry \n **☒ Loan Not Approved**")

        st.info(f"Model Accuracy: {accuracy * 100:.1f}%")
        logging.info("User requested model accuracy.")



























