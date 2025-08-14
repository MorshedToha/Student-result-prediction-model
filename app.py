import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
with open("best_student_performance_pipeline.joblib", 'rb') as f:
    model = joblib.load(f)

st.title("🎓 Student Performance Prediction")
st.write("Upload a CSV file with student academic data to get Pass/Fail predictions.")

# Upload CSV
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", data.head())

    # Make predictions
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    # Add results to dataframe
    data["Prediction"] = ["Pass" if p == 1 else "Fail" for p in predictions]
    data["Pass_Probability"] = probabilities.round(2)

    st.write("Predictions:", data)

    # Download predictions
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )




