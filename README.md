# Heart Disease Prediction App: Cardiovascular Risk Analysis

## Overview
This repository contains a Python web application designed to predict a patient's risk of having heart disease. By leveraging a pre-trained Random Forest machine learning model and an interactive Streamlit user interface, the application processes clinical and non-invasive test data to provide an instant, accurate cardiovascular risk classification.

While initial exploratory data analysis and model training evaluated multiple classifiers (where KNN achieved peak raw accuracy), a **Tuned Random Forest** was selected for deployment due to its superior clinical interpretability and ability to rank feature importance.

---

## Features
*   **Interactive Interface**: Built with Streamlit, allowing clinicians or users to easily input clinical data through a clean web form.
*   **Comprehensive Clinical Inputs**: Accepts standard medical parameters (Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina, ST Depression, Slope, Major Vessels, and Thalassemia).
*   **Automated Data Processing**: Categorical inputs are automatically processed in the backend using one-hot encoding, reindexing the data to strictly match the 18-column structure expected by the model. Missing categories are safely filled with zeros.
*   **Instant Risk Assessment**: Translates raw medical inputs into a clear binary diagnosis of either **"High Risk"** (presence of heart disease) or **"Low Risk"** (no heart disease).

---

## Dataset & Key Clinical Insights
The underlying model was trained on data from the official UCI Machine Learning Repository (Dataset ID: 45). Extensive Exploratory Data Analysis (EDA) on the 303 patient records revealed key clinical attributes that act as significant predictors for cardiovascular risk:

*   **`thalach` (Maximum Heart Rate Achieved)**: A critical indicator during physical stress tests; variations are strongly predictive of underlying heart conditions.
*   **`oldpeak` (ST Depression Induced by Exercise)**: Significant ST depression often indicates poor blood flow to the heart muscle, serving as a primary marker for coronary artery disease.
*   **`thal_7.0` (Reversible Defect)**: The presence of a reversible defect in thallium heart scans is a high-risk indicator.
*   **`cp_4` (Asymptomatic Chest Pain)**: Frequently identified by the model as a major red flag, even when the patient does not present typical symptoms.
*   **`ca` (Number of Major Vessels)**: The number of major vessels colored by fluoroscopy is a direct anatomical indicator of coronary health and disease severity.

---

## Repository Contents
*   **`app.py`** (or Main Streamlit Script): The Python script containing the frontend application UI and the data preprocessing pipelines using `pandas`.
*   **`heart_disease_rf_model.pkl`**: A serialized scikit-learn Random Forest Classifier loaded via `joblib` to compute the live predictions.
*   **`.gitignore`**: Configuration file excluding local development environments (`venv/`), secrets (`.env`), Python caches (`__pycache__/`), and OS-generated files.

---

## Installation and Usage

> **Note:** Ensure you have Python installed on your local machine before proceeding. 

**1. Clone the repository**
Clone this project to your local machine and navigate into the directory.

**2. Install dependencies**
Install the required libraries to run the application and load the model:
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

**3. Run the application**
Launch the Streamlit server to open the interactive web app in your default browser:
```bash
streamlit run app.py
```

**4. Make Predictions**
Use the sidebar or main interface to adjust the clinical parameters. Click the prediction button to evaluate the patient's cardiovascular risk profile based on the loaded Random Forest model.

##  Live Demo
Check out the live web application here:
