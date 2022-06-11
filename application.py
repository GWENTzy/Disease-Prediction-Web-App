from soupsieve import select
import streamlit as st 

from streamlit_option_menu import option_menu
from function import *
from home import home
from dataProcessing import dataProcessing
from aboutUs import aboutUs

# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

st.set_page_config(
    page_title="Disease Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
 )

selected = option_menu(None, ['Home', 'Data Processing', 'About us'], 
    icons=['house', 'book', 'list-task'], menu_icon="cast", default_index=0, orientation="horizontal")

if selected == 'Home':
    home()

if selected == 'Data Processing':
    dataProcessing()

if selected == 'About us':
    aboutUs()