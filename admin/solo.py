import streamlit as st 
import firebase_admin
from firebase_admin import storage, firestore
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd


firebase_admin.get_app()
db = firestore.client()
doc_ref = db.collection("users").document("test")
collec_ref = doc_ref.collection("scores")

df_front_balance = pd.DataFrame(columns=("Date","Image","Score"))
df_side_balance = pd.DataFrame(columns=("Date","Image","Score"))



for score in collec_ref.stream(): # the stream wil get both collections the side and front balance
    if score.id == "front balance":
        i = 1
        for key, value in score.to_dict().items():
            row = [key, value["image"], value["Scroe"]]
            df_front_balance.loc[i] = row
            print("Row:", row)
            i+=1
    elif score.id == "side balance":
        i = 1
        for key, value in score.to_dict().items():
            row = [key, value["image"], value["Scroe"]]
            df_side_balance.loc[i] = row
            print("Row:", row)
            i+=1



# Create tabs 
front_balance_tab, side_balance_tab = st.tabs(["Front Balance", "Side Balance"])
with front_balance_tab: 
    st_df = st.dataframe(df_front_balance,  column_config={
            "Image": st.column_config.ImageColumn(
                "Preview Image", help="Streamlit app preview screenshots",
                width= "large",
            )
        })
with side_balance_tab:
    st_df = st.dataframe(df_side_balance,  column_config={
        "Image": st.column_config.ImageColumn(
            "Preview Image", help="Streamlit app preview screenshots",
            width= "large",
        )
    })
