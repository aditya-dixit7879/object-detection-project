import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Load the model only once to save cloud server memory
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

st.set_page_config(page_title="Object Detection System", layout="wide")
st.title("Cloud-Based Live Webcam Detection")
st.write("This version uses WebRTC to securely access any user's camera over the internet.")

# WebRTC requires STUN servers to help devices find each other over the internet
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# This function acts as the "engine". It takes a frame from the user's browser,
# passes it to YOLO, and sends the drawn image back to the browser.
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert the browser's video frame into an image OpenCV can read
    img = frame.to_ndarray(format="bgr24")
    
    # Run YOLO object tracking on that specific frame
    results = model.track(img, persist=True)
    
    # Draw the bounding boxes and IDs
    annotated_img = results[0].plot()
    
    # Send the processed image back to the user's screen
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Create the WebRTC video player on the website
webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
