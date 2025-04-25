import streamlit as st
from evaluate import run_check
import firebase_admin
from firebase_admin import firestore, storage
from datetime import datetime
import cv2


def add_to_report(score, pos, image):

    now = datetime.now()
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app()
    db = firestore.client()
    doc_ref = db.collection("users").document(st.session_state.user_id)
    report_ref = doc_ref.collection("scores").document(pos)

    # s = f"{str(pos).replace(' ', '_')}_states"
    old_data = doc_ref.get().to_dict()[f"{str(pos).replace(' ', '_')}_states"]
    new_data = {
        "avg": (old_data["avg"] * old_data["count"] + score) / (old_data["count"] + 1) if old_data["avg"] > 0 else score,
        "max": max(old_data["max"], score ),
        "min": min(old_data["min"], score ),
        "last": score,
        "count": old_data["count"] + 1
    }
  

    # Adding  the IMAGES to storage
    try:
        # TODO save the bucket so its only created once 
        bucket = storage.bucket("forms-data-e0050.appspot.com")
        blob = bucket.blob(f'UsersData/EvaluatedImages/{st.session_state.user_id}{pos}{new_data["count"]}.png')
        blob.upload_from_string(image, content_type='image/png')
        blob.make_public()
        # url = blob.public_url
        report_ref.set({
        str(now):
        {
        "Scroe": score,
        "image": blob.public_url
        }
        }, merge=True)
        doc_ref.update({f"{pos.replace(' ', '_')}_states": new_data})
    except Exception as e:
        st.error(f"Error occurred while submiting your score, please try again!")
    else:
        st.success(f"score added to {pos} report")




data = st.session_state.user
name = data['name']
st.header(f'welcome back {name}')
poses = ["front balance", "side balance"]
pos = st.selectbox("evaluation for ...", poses, index=None, placeholder="Select pose ...")
image = st.file_uploader("Choose an image...", type=["jpg", "png"])


if image is not None:
    l = run_check(image, pos)
    if l:
        out_img, score, feedback = l
        test = st.image(out_img, channels='BGR')
        st.title(f"You scored {score}/10")
        if len(feedback) > 0:
            for line in feedback:
                st.write(line)

            st.page_link("user/reference.py", label="Go to References")

        is_success, buffer = cv2.imencode(".png", out_img)
        io_buf = buffer.tobytes()

        # Create a download button
        st.download_button(
            label="Download image",
            data=io_buf,
            file_name="image.png",
            mime="image/png"
        )
        st.button("submit", on_click=add_to_report, args=(score, pos, io_buf))
    else:
        st.error('invalid image, please upload another image!', icon="🚨")






