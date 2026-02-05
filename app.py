import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained pipeline
model = joblib.load("pipeline.pkl")

st.title("Employee Productivity Prediction")

st.header("Enter Employee Details:")

# Input fields
Age = st.number_input("Age", min_value=18, max_value=70, value=25)
Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
Department = st.selectbox("Department", ["HR", "IT", "Finance", "Marketing", "Operations"])
Designation = st.selectbox("Designation", ["Junior", "Senior", "Manager", "Director"])
ExperienceYears = st.number_input("Experience (Years)", min_value=0, max_value=50, value=2)
Skillset = st.multiselect(
    "Skillset",
    ["Python", "R", "SQL", "Machine Learning", "Data Analysis", "Communication", "Leadership"]
)
ProductivityScore = st.number_input("Productivity Score", min_value=0, max_value=100, value=50)
WorkLocation = st.selectbox("Work Location", ["Remote", "Onsite", "Hybrid"])
EducationLevel = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
LastPromotionYear = st.number_input("Last Promotion Year", min_value=2000, max_value=2026, value=2022)

# Predict button
if st.button("Predict"):
    # Prepare input data
    data = {
        "Age": Age,
        "Gender": Gender,
        "Department": Department,
        "Designation": Designation,
        "ExperienceYears": ExperienceYears,
        "Skillset": [Skillset],  # wrap in list for pipeline
        "ProductivityScore": ProductivityScore,
        "WorkLocation": WorkLocation,
        "EducationLevel": EducationLevel,
        "LastPromotionYear": LastPromotionYear
    }

    df = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(df)[0]

    # Show result
    st.success(f"Prediction Result: {prediction}")
