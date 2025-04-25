import math

import firebase_admin
import streamlit as st
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter


def check_valid_user(username_, password_):
    cred = credentials.Certificate(st.secrets["firebase"])
    try:
        firebase_admin.get_app()
    except ValueError as e:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    doc_ref = db.collection("users").document(username_)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        if password_ == data['password']:
            # name = data['name']
            st.session_state.user_id = username_
            st.session_state.user = data
            st.session_state.role = data['role']
            return True
            # st.switch_page("pages/app.py")
            # st.success(f' valid user with name {name} ').

        else:
            st.error("invalid password")
    else:
        st.error("invalid username")
    return False


def sign_up(username, name, password):
    poses = ["front_balance", "side_balance"]
    states = {
        "avg": 0,
        "max": -1,
        "min": 20,
        "last": 0,
        "count": 0
    }
    data = {
        "name": name,
        "password": password,
        "role": "user",
    }
    for i in poses:
        data[f'{i}_states'] = states

    cred = credentials.Certificate(st.secrets["firebase"])
    try:
        firebase_admin.get_app()
    except ValueError as e:
        print(e)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    if db.collection("users").where(filter=FieldFilter("name", "==", username)).get():
        st.error("Invalid username")
        return False
    else:
        try:
            db.collection("users").document(username).set(data)
            st.session_state.user_id = username
            st.session_state.user = data
            st.session_state.role = data['role']
            return True
        except ValueError:
            st.error("An error occurred please try again!")
            return False





