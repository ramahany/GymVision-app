import streamlit as st
import mediapipe as mp
from functions import *

if "role" not in st.session_state:
    st.session_state.role = None
if "state" not in st.session_state:
    st.session_state.state = "logging in"
ROLES = [None, "user", "admin"]


def login():

    if st.session_state.state == "logging in":

       
        st.title("Welcome to GymVision")
        st.header("Login:")
        username = st.text_input(label="User or Email:", value="")
        password = st.text_input(label="password:", value="", type="password")

        if st.button(label="login", type="primary"):
            if check_valid_user(username, password):
                st.rerun()
        if st.button(label="Create account", type="tertiary"):
            st.session_state.state = "signing up"
            st.rerun()

    if st.session_state.state == "signing up":
        st.title("Welcome to GymVision")
        st.header("Signup:")
        username = st.text_input(label="User_name or Email:", value="")
        name = st.text_input(label="name", value="")
        password = st.text_input(label="password:", value="", type="password")

        if st.button(label="signup", type="primary") and username != '' and password != '' and name != '':
            if sign_up(username, name, password):
                st.rerun()

        if st.button(label="already have an account?", type="tertiary"):
            st.session_state.state = "logging in"
            st.rerun()


def logout():
    for key in st.session_state.keys():
        del st.session_state[key]
    # st.session_state.role = None
    # st.session_state.data = None
    st.rerun()


role = st.session_state.role

logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

# User pages
reference = st.Page("user/reference.py", title="Reference", icon=":material/book_4_spark:")
user_p = st.Page(
    "user/user.py",
    title="Profile",
    icon=":material/person:",
    default=(role == "user"),
)
user_p2 = st.Page(
    "user/app.py",
    title="evaluation page",
    icon=":material/published_with_changes:")

# Admin pages
data_pg = st.Page(
    "admin/admin.py",
    title="admin page",
    icon=":material/person:",
    default=(role == "admin"),
)
add_admin_pg = st.Page(
    "admin/addingUser.py",
    title="Add Admin",
    icon=":material/person_add:",

)
solo_page = st.Page(
    "admin/solo.py",
    title="Solo",
    icon=":material/person_add:",

)


account_pages = [logout_page]

user_pages = [user_p, reference, user_p2]
admin_pages = [data_pg, add_admin_pg, solo_page]

st.logo("images/logo.webp",  size="large", icon_image="images/logo.webp")

page_dict = {}
if st.session_state.role == "user":
    page_dict["user"] = user_pages

if st.session_state.role == "admin":
    page_dict["Admin"] = admin_pages

if len(page_dict) > 0:
    pg = st.navigation(page_dict | {"Account": account_pages})
else:
    pg = st.navigation([st.Page(login)])

pg.run()
