import streamlit as st
import joblib
import pandas as pd

# Loading the saved model
model = joblib.load('heart_disease_rf_model.pkl')

# Setting up the streamlit interface
st.title("Heart Disease Prediction App")
st.write("Enter the patient's clinical data below to predict the risk of Heart Disease.")

# Creating input fields for standard features
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pan Type (0-3)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)

with col2:
    chol = st.number_input("Serum cholestorol (mg/dl)", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", value=150)

with col3:
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", value=0.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1, 3, 6, 7)", options=[1, 3, 6, 7])

#Creating prediction button
if st.button("Predict Heart Disease Risk"):

    input_data = pd.DataFrame({
      'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
      'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
      'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope],
      'ca': [ca], 'thal': [thal]  
    })

    input_data_encoded = pd.get_dummies(input_data, columns=['cp', 'restecg', 'slope', 'thal'])

    model_columns = [
        'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
        'cp_2', 'cp_3', 'cp_4', 'restecg_1', 'restecg_2', 'slope_2', 'slope_3', 'thal_6.0', 'thal_7.0'
    ]

    input_data_final = input_data_encoded.reindex(columns=model_columns, fill_value=0)


    prediction = model.predict(input_data_final)

#Displaying the results
    if prediction[0] == 1:
        st.error("High Risk: The Model predicts the Presence of Heart Disease.")
    else:
        st.success("Low Risk: The Model predicts NO Heart Disease.")
