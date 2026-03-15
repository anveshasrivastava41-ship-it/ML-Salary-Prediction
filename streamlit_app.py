import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("💼 Salary Prediction ML App")

st.write("Enter job details to predict salary")

experience = st.slider("Experience Level (0-3)", 0, 3)
employment = st.slider("Employment Type (0-3)", 0, 3)
job_title = st.slider("Job Title Code (0-100)", 0, 100)
location = st.slider("Company Location Code (0-100)", 0, 100)
company_size = st.slider("Company Size (0-2)", 0, 2)

if st.button("Predict Salary"):

    input_data = np.array([[experience, employment, job_title, location, company_size]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Salary: ${int(prediction[0])}")

skills = st.multiselect(
    "Select Skills",
    ["Python","Machine Learning","SQL","Deep Learning","Data Analysis","AWS","TensorFlow"]
)
skill_score = len(skills)
input_data = np.array([[experience, employment, job_title, location, company_size, skill_score]])