import streamlit as st

from web_functions import predict

def app(df, x, y):

    st.title("Halaman Prediksi")

    col1, col2 = st.columns(2)
    with col1 :
        age  = st.number_input("input nilai age")
    with col1 :
        sex = st.number_input("input nilai sex")
    with col1 :
        cp = st.number_input("input nilai cp")
    with col1 :
        trtbps = st.number_input("input nilai trtbps")
    with col1 :
        chol = st.number_input("input nilai chol")
    with col1 :
        fbs = st.number_input("input nilai fbs")
    with col1 :  
        restecg = st.number_input("input nilai restecg")
    with col2 : 
        thalachh = st.number_input("input nilai thalachh")
    with col2 :
        exng = st.number_input("input nilai exng")
    with col2 :
        oldpeak = st.number_input("input nilai oldpeak")
    with col2 :
        slp = st.number_input("input nilai slp")
    with col2 :
        caa = st.number_input("input nilai caa")
    with col2 :
        thall = st.number_input("input nilai thall")

    features = [age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]

    # tombol prediksi
    if st.button("Prediksi"):
        prediction, score = predict(x,y,features)
        score = score
        st.info("Prediksi Sukses...")

        if(prediction == 1):
            st.warning("Pasien terkena serangan jantung")
        else:
            st.success("Pasien tidak terkena serangan jantung")

        st.write("Model yang digunakan memiliki tingkat akurasi", (score*100),"%")