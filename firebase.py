import streamlit
from functions import *


st.title("Try login:")
username = st.text_input(label="User or Email:", value="")
password = st.text_input(label="password:", value="", type="password")

if st.button(label="login", type="primary"):
    check_valid_user(username, password)

if st.button(label="SignUp", type="tertiary"):
    streamlit.switch_page("pages/sign_up.py")







