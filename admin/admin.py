import streamlit as st
from admin_functions import get_user_data

data = st.session_state.user
name = data['name']
st.title(f'Welcome Back {name}')
st.header('User Data')

users = get_user_data()
print(users)
st.dataframe(users)