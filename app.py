import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

# Load the YOLOv8 AI model
model = YOLO('yolov8n.pt')

st.set_page_config(page_title="Object Detection & Tracking", layout="wide")
st.title("Real-Time Object Tracking System")
st.write("This advanced version assigns unique IDs to objects across frames.")

st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Level", 0.0, 1.0, 0.5)

# Give users a choice: Webcam (Local) or Video Upload (Cloud)
source_radio = st.sidebar.radio("Select Source", ["Webcam", "Upload Video"])

st.subheader("Tracking Feed")
video_placeholder = st.empty()

if source_radio == "Webcam":
    run_camera = st.checkbox("Start Camera")
    if run_camera:
        cap = cv2.VideoCapture(0)
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not found. Please use 'Upload Video' if you are on the cloud.")
                break
            
            # PHASE 2 UPGRADE: Using .track() instead of just calling the model
            # persist=True keeps the IDs consistent across frames
            results = model.track(frame, conf=confidence, persist=True)
            
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame, channels="RGB")
        cap.release()

elif source_radio == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file (mp4, avi)", type=['mp4', 'avi'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # PHASE 2 UPGRADE: Using .track() for uploaded videos too
            results = model.track(frame, conf=confidence, persist=True)
            
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame, channels="RGB")
        cap.release()
