import streamlit as st
import mediapipe as mp
import firebase_admin
from firebase_admin import firestore, storage
from datetime import datetime
import cv2
import numpy as np


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

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Calculations
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def draw_two_lines(p1, p2, p3, image, color):
    cv2.line(image, tuple(np.multiply([p1[0], p1[1]], [image.shape[1], image.shape[0]]).astype(int)),
             tuple(np.multiply([p2[0], p2[1]], [image.shape[1], image.shape[0]]).astype(int))
             , color, 2)
    cv2.line(image, tuple(np.multiply([p2[0], p2[1]], [image.shape[1], image.shape[0]]).astype(int)),
             tuple(np.multiply([p3[0], p3[1]], [image.shape[1], image.shape[0]]).astype(int))
             , color, 2)


def check_valid_angles(points, image, min_, max_, landmarks):
    p1 = [landmarks[points[0]].x, landmarks[points[0]].y]
    joint = [landmarks[points[1]].x, landmarks[points[1]].y]
    p2 = [landmarks[points[2]].x, landmarks[points[2]].y]
    angle = calculate_angle(p1, joint, p2)
    # cv2.putText(image, str(int(angle)),
    #             tuple(np.multiply(joint, [image.shape[1], image.shape[0]]).astype(int)),
    #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    score = 0
    if min_ <= angle <= max_:
        draw_two_lines(p1, joint, p2, image, (0, 255, 0))
        score = 1
    else:
        draw_two_lines(p1, joint, p2, image, (0, 0, 255))
    return angle, score


def check_shoulder(landmarks, image):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    check_x = True if (abs(left_shoulder[0] - right_shoulder[0]) < 0.05) else False
    check_y = True if (abs(left_shoulder[1] - right_shoulder[1]) < 0.05) else False
    color = (0, 255, 0) if (check_x and check_y) else (0, 0, 255)
    cv2.line(image, tuple(np.multiply(left_shoulder, [image.shape[1], image.shape[0]]).astype(int)),
             tuple(np.multiply(right_shoulder, [image.shape[1], image.shape[0]]).astype(int))
             , color, 2)
    return 1 if (check_x and check_y) else 0


def check_hip(landmarks, image):

    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]

    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]

    check_x = True if (abs(left_hip[0]-right_hip[0]) < 0.06) else False
    check_y = True if (abs(left_hip[1]-right_hip[1]) < 0.06) else False

    color = (0, 255, 0) if (check_x and check_y) else (0, 0, 255)

    cv2.line(image, tuple(np.multiply([left_hip[0], left_hip[1]], [image.shape[1], image.shape[0]]).astype(int)),
             tuple(np.multiply([right_hip[0], right_hip[1]], [image.shape[1], image.shape[0]]).astype(int))
             , color, 2)
    return 1 if (check_x and check_y) else 0


def get_angle_between_legs(landmarks, img):
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    center_hip = [(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x)/2,
                  (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)/2]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    angle = calculate_angle(right_knee, center_hip, left_knee)
    color = (10, 255, 0) if angle >= 90 else (10, 0, 255)
    draw_two_lines(right_knee, center_hip, left_knee, img, color)
    score = 1 if angle >= 90 else 0
    return angle, score


def check_elbows(landmarks, img):
    score = 0
    left_elbow_points = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
    ]
    right_elbow_index = mp_pose.PoseLandmark.RIGHT_ELBOW.value

    if landmarks[right_elbow_index].visibility > 0.1:
        index = mp_pose.PoseLandmark.RIGHT_WRIST.value
        cv2.line(img, tuple(np.multiply([landmarks[right_elbow_index].x, landmarks[right_elbow_index].y], [img.shape[1], img.shape[0]]).astype(int)),
                 tuple(np.multiply([landmarks[index].x, landmarks[index].y], [img.shape[1], img.shape[0]]).astype(int))
                 , (10, 255, 0), 2)
    else:
        score += 1
    x, s = check_valid_angles(left_elbow_points, img, 120, 190, landmarks)
    print(f"angle between elbow = {x} ")
    score += s
    return score


def analyse_pic_front_balance(landmarks, img):
    total_score = 0
    feedback = []
    right_knee_points = [
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    left_knee_points = [
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
    ]

    right_hip_points = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
    ]

    left_hip_points = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
    ]

    left_ankle_points = [
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
    ]

    # for right leg
    right_knee_angle, score = check_valid_angles(right_knee_points, img, 170, 185, landmarks)
    total_score += score

    if score == 0:
        feedback.append('your right knee should be straight')

    left_knee_angle, score = check_valid_angles(left_knee_points, img, 170, 185, landmarks)
    total_score += score

    if score == 0:
        feedback.append('your left knee should be straight')

    left_hip_angle, score = check_valid_angles(left_hip_points, img, 100, 190, landmarks)
    total_score += score

    if score == 0:
        feedback.append('your left knee should be straight')

    left_ankle_angle, score = check_valid_angles(left_ankle_points, img, 130, 180, landmarks)
    total_score += score
    print(f'angle ankle hip: {left_ankle_angle}')

    score = check_hip(landmarks, img)
    total_score += score
    score = check_shoulder(landmarks, img)
    total_score += score
    angle, score = get_angle_between_legs(landmarks, img)
    total_score += score
    if score == 1:
        right_hip_angle, score = check_valid_angles(right_hip_points, img, 85, 120, landmarks)
    else:
        right_hip_angle, score = check_valid_angles(right_hip_points, img, 90, 105, landmarks)

    total_score += score
    print(f'angle between hip: {right_hip_angle}')
    score = check_elbows(landmarks, img)
    total_score += score
    return img, total_score, feedback


def run_check(img_path, pose_chosen):
    bytes_data = img_path.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    if img.shape[0] > 900:
        img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(input_img)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
        else :
            return None
        if pose_chosen == "front balance":
            out_put, score, feed_back = analyse_pic_front_balance(landmarks, img)
        elif pose_chosen == "side balance":
            # TODO finish the side balance
            feed_back = ''
            out_put, score = input_img, 8

    return out_put, score, feed_back





