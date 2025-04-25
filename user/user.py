import streamlit as st
import firebase_admin
from firebase_admin import firestore
import pandas as pd
import numpy as np

front_balance_data , side_balance_data = {}, {}

firebase_admin.get_app()
db = firestore.client()
doc_ref = db.collection("users").document(st.session_state.user_id)
collec_ref = doc_ref.collection("scores")

if collec_ref:
    for doc in collec_ref.stream():
        if doc.id == "front balance":
            front_balance_data = doc.to_dict()

        elif doc.id == "side balance":
            side_balance_data = doc.to_dict()

data = st.session_state.user
st.header(f'Welcome {data["name"]}')


# Front balance graph and data
st.title("Front Balance Graph")
df = pd.DataFrame({
    'time': front_balance_data.keys(),
    'score': [inner["Scroe"] for inner in front_balance_data.values()]
})
# df['time'] = pd.to_datetime(df['time'])  # Convert to datetime if needed

st.line_chart(df, x='time', y='score')

# side balance graph and data
st.title("Side Balance Graph")

df = pd.DataFrame({
    'time': side_balance_data.keys(),
    'score': [inner["Scroe"] for inner in side_balance_data.values()]
})

# df['time'] = pd.to_datetime(df['time'])  # Convert to datetime if needed

st.line_chart(df, x='time', y='score')