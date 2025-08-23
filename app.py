import streamlit as st
import streamlit.components.v1 as stc
import pickle

# === Load Model ===
model_filename = "logistic_regression_model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

st.title("Customer Segmentation Prediction 🚀")
st.write("Aplikasi ini menggunakan **Logistic Regression** untuk memprediksi segmentasi customer.")

# === Input User ===
st.sidebar.header("Input Customer Data")

# contoh input (sesuaikan dengan kolom datasetmu!)
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
annual_income = st.sidebar.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
spending_score = st.sidebar.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Buat dataframe dari input user
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [1 if gender == "Male" else 0],  # contoh encoding
    "Annual_Income": [annual_income],
    "Spending_Score": [spending_score]
})

st.subheader("Input Data")
st.write(input_data)

# === Prediksi ===
if st.button("Prediksi Segmentation"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    st.success(f"Hasil Prediksi: **Segment {prediction[0]}**")
    st.write("Probabilitas Tiap Segmen:")
    st.write(pd.DataFrame(proba, columns=[f"Segment {i}" for i in range(proba.shape[1])]))
