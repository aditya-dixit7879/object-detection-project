import streamlit as st
import cv2
import tempfile
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Load the YOLO model once to save cloud server memory
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

st.set_page_config(page_title="Object Detection System", layout="wide")
st.title("Advanced Object Detection & Tracking")
st.write("Choose between live webcam tracking or uploading a pre-recorded video.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Level", 0.0, 1.0, 0.5)

# The missing radio button is back!
source_radio = st.sidebar.radio("Select Source", ["Live Webcam", "Upload Video"])

# --- OPTION 1: LIVE WEBCAM (WebRTC) ---
if source_radio == "Live Webcam":
    st.subheader("Live Cloud Webcam (WebRTC)")
    st.write("Click 'START' to grant camera access and begin tracking.")
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        # Convert browser frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")
        
        # Run YOLO tracking
        results = model.track(img, conf=confidence, persist=True)
        annotated_img = results[0].plot()
        
        # Send processed frame back to browser
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# --- OPTION 2: UPLOAD VIDEO ---
elif source_radio == "Upload Video":
    st.subheader("Process a Video File")
    uploaded_file = st.file_uploader("Upload a video file (mp4, avi)", type=['mp4', 'avi'])
    
    if uploaded_file is not None:
        video_placeholder = st.empty()
        
        # Save file temporarily on the cloud server to read it
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLO tracking on the video
            results = model.track(frame, conf=confidence, persist=True)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display it on the website
            video_placeholder.image(annotated_frame, channels="RGB")
            
        cap.release()
