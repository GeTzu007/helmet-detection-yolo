import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

from utils.detector import detect_frame

# PAGE CONFIG
st.set_page_config(
    page_title="AI Helmet Detection",
    page_icon="🦺",
    layout="wide"
)

# TITLE
st.title("🦺 AI Helmet Detection System")

st.markdown("""
Realtime Helmet Detection menggunakan YOLOv8
""")

# SIDEBAR
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Webcam", "Image", "Video"]
)

confidence = st.sidebar.slider(
    "Confidence",
    0.1,
    1.0,
    0.5
)

# OUTPUT FOLDER
os.makedirs("outputs", exist_ok=True)

# =========================================
# WEBCAM MODE
# =========================================

if mode == "Webcam":

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    fps_text = st.empty()
    detect_text = st.empty()

    camera = cv2.VideoCapture(0)

    while run:

        success, frame = camera.read()

        if not success:
            st.error("Camera Error")
            break

        result_frame, fps, total = detect_frame(
            frame,
            confidence
        )

        FRAME_WINDOW.image(
            result_frame,
            use_container_width=True
        )

        fps_text.markdown(f"### FPS: {fps:.2f}")

        detect_text.markdown(
            f"### Total Detection: {total}"
        )

    camera.release()

# =========================================
# IMAGE MODE
# =========================================

elif mode == "Image":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image)

        image_np = np.array(image)

        result_frame, fps, total = detect_frame(
            image_np,
            confidence
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(image)

        with col2:
            st.subheader("Detection")
            st.image(result_frame)

        st.success(f"Detection: {total}")

# =========================================
# VIDEO MODE
# =========================================

elif mode == "Video":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        temp_video = tempfile.NamedTemporaryFile(delete=False)

        temp_video.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video.name)

        FRAME_WINDOW = st.image([])

        fps_text = st.empty()

        detect_text = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            result_frame, fps, total = detect_frame(
                frame,
                confidence
            )

            FRAME_WINDOW.image(
                result_frame,
                use_container_width=True
            )

            fps_text.markdown(f"### FPS: {fps:.2f}")

            detect_text.markdown(
                f"### Detection: {total}"
            )

        cap.release()

# FOOTER
st.sidebar.markdown("---")

st.sidebar.info("""
Built With:
- Streamlit
- YOLOv8
- OpenCV
- Python
""")