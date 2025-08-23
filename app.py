import streamlit as st
import streamlit.components.v1 as stc
import pickle

# === Load Model ===
model_filename = "logistic_regression_model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.title("Customer Segmentation Prediction 🚀")
import streamlit as st
import pickle
import pandas as pd

# =====================
# Load model
# =====================
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# =====================
# Title
# =====================
st.title("Customer Segmentation Prediction")
st.write("Masukkan data pelanggan untuk memprediksi segmentasi.")

# =====================
# Input Form
# =====================
gender = st.selectbox("Gender", ["Male", "Female"])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
graduated = st.selectbox("Graduated", ["Yes", "No"])
work_experience = st.number_input("Work Experience (tahun)", min_value=0, max_value=50, step=1)
family_size = st.number_input("Family Size", min_value=1, max_value=20, step=1)
var_1 = st.selectbox("Var_1", ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6"])

profession = st.selectbox(
    "Profession",
    ["Artist", "Doctor", "Engineer", "Entertainment", "Executive",
     "Healthcare", "Homemaker", "Lawyer", "Marketing"]
)

spending_score = st.selectbox("Spending Score", ["Average", "High", "Low"])
age_category = st.selectbox("Age Category", ["Remaja", "Dewasa", "Lansia"])

# =====================
# Encoding Input
# =====================
# Mapping categorical ke 0/1
gender = 1 if gender == "Male" else 0
ever_married = 1 if ever_married == "Yes" else 0
graduated = 1 if graduated == "Yes" else 0

# One-hot untuk profession
profession_dict = {
    "Profession_Artist": 1 if profession == "Artist" else 0,
    "Profession_Doctor": 1 if profession == "Doctor" else 0,
    "Profession_Engineer": 1 if profession == "Engineer" else 0,
    "Profession_Entertainment": 1 if profession == "Entertainment" else 0,
    "Profession_Executive": 1 if profession == "Executive" else 0,
    "Profession_Healthcare": 1 if profession == "Healthcare" else 0,
    "Profession_Homemaker": 1 if profession == "Homemaker" else 0,
    "Profession_Lawyer": 1 if profession == "Lawyer" else 0,
    "Profession_Marketing": 1 if profession == "Marketing" else 0,
}

# One-hot untuk Spending Score
spending_dict = {
    "Spending_Score_Average": 1 if spending_score == "Average" else 0,
    "Spending_Score_High": 1 if spending_score == "High" else 0,
    "Spending_Score_Low": 1 if spending_score == "Low" else 0,
}

# One-hot untuk Age Category
age_dict = {
    "Age_Category_Remaja": 1 if age_category == "Remaja" else 0,
    "Age_Category_Dewasa": 1 if age_category == "Dewasa" else 0,
    "Age_Category_Lansia": 1 if age_category == "Lansia" else 0,
}

# =====================
# Buat DataFrame sesuai fitur model
# =====================
input_data = pd.DataFrame([{
    "Gender": gender,
    "Ever_Married": ever_married,
    "Graduated": graduated,
    "Work_Experience": work_experience,
    "Family_Size": family_size,
    "Var_1": var_1,  # kalau var_1 di training di-encode, pastikan mapping dulu

    # One-hot profession
    **profession_dict,

    # Spending Score
    **spending_dict,

    # Age Category
    **age_dict,
}])

st.write("Data yang diproses:")
st.dataframe(input_data)

# =====================
# Prediksi
# =====================
if st.button("Prediksi Segmentasi"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("Hasil Prediksi")
    st.write(f"**Segmentasi:** {prediction}")

    st.subheader("Probabilitas Tiap Kelas")
    for i, prob in enumerate(prediction_proba):
        st.write(f"Segment {i}: {prob:.4f}")

