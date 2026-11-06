import os
import requests
import streamlit as st

st.set_page_config(page_title="Churn Prediction", layout='centered')

API_URL = os.getenv("API_URL", 'http://localhost:8000/predict')
st.title("Cutomer Churn Prediction")

with st.form('churn form'):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
        total = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * 70.0), step=10.0)

    submitted = st.form_submit_button('Predict', use_container_width=True)

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        if response.status_code ==200:
            data = response.json()
            prob=data['churn_probability']
            prediction = data['prediction']
            risk = data['risk_level']

            st.divider()

            st.metric(label='Churn Probability', value=f'{prob * 100:.1f}%')

            if risk=='High':
                st.error(f'Prediction: {risk} Risk of Churn')
            else:
                st.success(f'Prediction: {risk} Risk of Churn')
        else:
            st.error(f'Error {response.status_code}: {response.text}')
    except requests.exceptions.RequestException as e:
        st.error(f'API connection failed: {e}')

        