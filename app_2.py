import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title("Aplikasi Prediksi Stroke")

gender = st.selectbox("Apa Jenis Kelamin Anda?", ['Male', 'Female'])
ever_married = st.selectbox("Apakah Anda Sudah Menikah?", ['Yes', 'No'])
work_type = st.selectbox("Apa Pekerjaan Anda?", ['Self-employed', 'Private', 'Govt_job', 'children'])
Residence_type = st.selectbox("Bagaimana Status Permukiman Tempat Tinggal Anda?", ['Rural', 'Urban'])
smoking_status = st.selectbox("Apakah Anda Seorang Perokok?", ['never smoked', 'Unknown', 'formerly smoked', 'smokes'])
age = st.number_input("Berapa Umur Anda?")
hypertension = st.number_input("Apakah Anda Pernah Mengidap Hipertensi? Ketik 1 untuk ya, 0 untuk tidak")
heart_disease = st.number_input("Apakah Anda Punya Penyakit Hati? Ketik 1 untuk ya, 0 untuk tidak")
avg_glucose_level = st.number_input("Berapa Level Rata-Rata Gula Anda?")
bmi = st.number_input("Berapa Indeks Massa Tubuh Anda?")


model = pickle.load(open("model.pkl","rb"))

data_input = pd.DataFrame([[gender, ever_married, work_type, Residence_type, smoking_status,
                            age, hypertension, heart_disease, avg_glucose_level, bmi
]],
            columns=["gender","ever_married","work_type","Residence_type","smoking_status",
                     "age","hypertension","heart_disease","avg_glucose_level","bmi"
])


hasil = model.predict(data_input)

if st.button('Submit'):
    result = model.predict(data_input)[0]
    if result == 1:
        st.text('Anda diprediksi mengidap penyakit stroke.')
    else:
        st.text('Anda diprediksi tidak mengidap penyakit stroke.')