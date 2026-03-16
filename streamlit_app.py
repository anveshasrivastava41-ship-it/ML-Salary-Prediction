import streamlit as st
import pickle
import numpy as np
st.set_page_config(
    page_title="Paylytics",
    page_icon="💰",
    layout="wide"
)
exchange_rates = {
    "USD": 1,
    "INR": 83,
    "EUR": 0.92,
    "GBP": 0.78
}
# Load model
model = pickle.load(open("model.pkl", "rb"))
job_encoder = pickle.load(open("job_encoder.pkl", "rb"))
location_encoder = pickle.load(open("location_encoder.pkl", "rb"))
employment_encoder = pickle.load(open("employment_encoder.pkl", "rb"))
size_encoder = pickle.load(open("size_encoder.pkl", "rb"))
experience_encoder = pickle.load(open("experience_encoder.pkl","rb"))

st.title("💼 PAYLYTICS")
st.write("Enter job details to predict salary")

# Experience mapping
experience_map = {
    "Entry Level": "EN",
    "Mid Level": "MI",
    "Senior": "SE",
    "Executive": "EX"
}

experience_label = st.selectbox(
    "Experience Level",
    list(experience_map.keys())
)

experience = experience_map[experience_label]


# Employment mapping
employment_map = {
    "Full-time": "FT",
    "Part-time": "PT",
    "Contract": "CT",
    "Freelance": "FL"
}

employment_label = st.selectbox(
    "Employment Type",
    list(employment_map.keys())
)

employment = employment_map[employment_label]


job_title = st.selectbox("Job Title", job_encoder.classes_)
location = st.selectbox("Company Location", location_encoder.classes_)

company_size = st.selectbox(
    "Company Size",
    ["S","M","L"]
)
skills = st.multiselect(
    "Select Skills",
    ["Python","Machine Learning","SQL","Deep Learning","Data Analysis","AWS","TensorFlow"]
)
skill_score = len(skills)

currency = st.selectbox(
    "Select Currency",
    ["USD", "INR", "EUR", "GBP"]
)

if st.button("Predict Salary"):

    try:
        experience_code = experience_encoder.transform([experience])[0]
        employment_code = employment_encoder.transform([employment])[0]
        job_title_code = job_encoder.transform([job_title])[0]
        location_code = location_encoder.transform([location])[0]
        size_code = size_encoder.transform([company_size])[0]

        input_data = np.array([[experience_code,
                                employment_code,
                                job_title_code,
                                location_code,
                                size_code]])

        prediction = model.predict(input_data)
        salary_usd = prediction[0]
        converted_salary = salary_usd * exchange_rates[currency]
        st.success(f"Estimated Salary: {currency} {int(converted_salary)}")

    except:
        st.error("One of the inputs is not present in the dataset. Please enter a valid value.")







