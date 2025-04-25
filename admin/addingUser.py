import streamlit as st
from admin_functions import add_admin

st.header("Add Admin:")
username = st.text_input(label="User_name or Email:", value="")
name = st.text_input(label="name", value="")
password = st.text_input(label="password:", value="", type="password")

if st.button(label="Add Admin", type="primary") and username != '' and password != '' and name != '':
    flag, massage = add_admin(username, name, password)
    if flag:
        st.success(massage)
    else:
        st.error(massage)
    
