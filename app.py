import streamlit as st
import cv2
from ultralytics import YOLO

# Load the YOLOv8 AI model
# It will download a small file named 'yolov8n.pt' automatically the first time
model = YOLO('yolov8n.pt')

# Set the title and layout of the website
st.set_page_config(page_title="Object Detection System", layout="wide")

# Main heading
st.title("Real-Time Object Detection & Tracking")
st.write("This project uses YOLOv8 for real-time object detection.")

# Create a sidebar for settings and controls
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Level", 0.0, 1.0, 0.5)

st.subheader("Camera Feed")

# Create a checkbox to turn the camera on and off safely
run_camera = st.checkbox("Start Camera")

# Placeholder to display the video feed
video_placeholder = st.empty()

# Open the webcam (0 means the default laptop camera)
cap = cv2.VideoCapture(0)

# Loop to continuously read frames from the camera
while run_camera:
    ret, frame = cap.read()
    
    if not ret:
        st.error("Failed to capture video from camera.")
        break
        
    # Run YOLO detection on the current frame
    # We pass the confidence level selected from the sidebar
    results = model(frame, conf=confidence)
    
    # YOLO has a built-in feature to draw the boxes and labels on the image
    annotated_frame = results[0].plot()
    
    # OpenCV reads images in BGR format, but web browsers need RGB format
    # We convert the colors so they look natural
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    # Display the updated image on the website
    video_placeholder.image(annotated_frame, channels="RGB")

# Turn off the camera when the checkbox is unchecked
cap.release()