import streamlit as st
import pickle
import pandas as pd

# =====================
# Load model & feature names
# =====================
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# =====================
# Title
# =====================
st.title("Customer Segmentation Prediction 🚀")
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

age_category = st.selectbox(
    "Age Category",
    ["Remaja", "Dewasa", "Lansia"],
    help="Kategori umur pelanggan:\n- Remaja (13–25 tahun)\n- Dewasa (26–45 tahun)\n- Lansia (46 tahun ke atas)"
)

# Tambahkan penjelasan langsung di bawah form agar user mobile juga paham
st.caption("ℹ️ **Remaja:** 13–25 tahun | **Dewasa:** 26–45 tahun | **Lansia:** 46 tahun ke atas")

# =====================
# Encoding Input
# =====================
# Label Encoding manual
gender = 1 if gender == "Male" else 0
ever_married = 1 if ever_married == "Yes" else 0
graduated = 1 if graduated == "Yes" else 0

# Mapping Var_1 sesuai label encoding
var1_mapping = {
    "Cat_1": 0,
    "Cat_2": 1,
    "Cat_3": 2,
    "Cat_4": 3,
    "Cat_5": 4,
    "Cat_6": 5,
}
var_1 = var1_mapping[var_1]

# One-hot Profession
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

# One-hot Spending Score
spending_dict = {
    "Spending_Score_Average": 1 if spending_score == "Average" else 0,
    "Spending_Score_High": 1 if spending_score == "High" else 0,
    "Spending_Score_Low": 1 if spending_score == "Low" else 0,
}

# One-hot Age Category
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
    "Var_1": var_1,

    **profession_dict,
    **spending_dict,
    **age_dict,
}])

# Reindex agar sesuai fitur training
input_data = input_data.reindex(columns=feature_names, fill_value=0)

st.write("🔎 Data yang diproses ke model:")
st.dataframe(input_data)

# =====================
# Prediksi
# =====================
if st.button("Prediksi Segmentasi"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("🎯 Hasil Prediksi")
    st.write(f"**Segmentasi:** {prediction}")

    st.subheader("📊 Probabilitas Tiap Kelas")
    for i, prob in enumerate(prediction_proba):
        st.write(f"Segment {i}: {prob:.4f}")

