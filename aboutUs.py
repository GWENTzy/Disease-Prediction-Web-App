import streamlit as st

from PIL import Image

def aboutUs():
    st.markdown("<h2 style='text-align: center; color: white;'>About Us</h2>", unsafe_allow_html=True)
    st.markdown('<br />', unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: white;'>Masa pandemi COVID-19 membuat banyak orang menjadi mudah khawatir dengan kondisi kesehatan mereka. Banyak dari mereka yang merasa bosan dan malas untuk mengantri di rumah sakit, klinik, atau puskesmas. Selain karena waktu antri yang lama, hal ini juga dapat disebabkan orang tersebut terpapar virus dari orang lain. Aplikasi kami ada untuk menjawab solusi dari permasalahan tersebut dengan memberikan prediksi penyakit yang diderita hanya dengan memberikan gejala - gejala yang dialami user</h5>", unsafe_allow_html=True)
    
    st.markdown('<br />', unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: white;'>Our Team</h2>", unsafe_allow_html=True)
    st.markdown('<br />', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        image = Image.open('images/alvon.png')    
        st.image(image, use_column_width=True)
        st.markdown("<h5 style='text-align: center; color: white;'>Alvon Danilo Sukardi</h5>", unsafe_allow_html=True)
    with col2:
        image = Image.open('images/Gerry.png')
        st.image(image, use_column_width=True)
        st.markdown("<h5 style='text-align: center; color: white;'>Gerry Wiliam Nahlohy</h5>", unsafe_allow_html=True)
    with col3:
        image = Image.open('images/michelle.png')
        st.image(image, use_column_width=True)
        st.markdown("<h5 style='text-align: center; color: white;'>Misell Christian Bell</h5>", unsafe_allow_html=True)
