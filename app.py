import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained pipeline
model = joblib.load("best_student_performance_pipeline.joblib")

st.title("🎓 Student Performance Prediction App")
st.write("Enter the details below to predict student performance.")

# Collect user inputs
gender = st.selectbox("Gender", ["male", "female"])
race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_education = st.selectbox(
    "Parental Level of Education",
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
)
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])

math_score = st.number_input("Math Score", min_value=0, max_value=100, value=70)
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=70)

if st.button("Predict Performance"):
    # Prepare input data
    input_data = pd.DataFrame({
        "gender": [gender],
        "race_ethnicity": [race_ethnicity],
        "parental_level_of_education": [parental_education],
        "lunch": [lunch],
        "test_preparation_course": [test_preparation_course],
        "math_score": [math_score],
        "reading_score": [reading_score],
        "writing_score": [writing_score]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Performance Level: **{prediction}**")




