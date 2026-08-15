import streamlit as st
import cv2
import numpy as np
import av
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_webrtc import webrtc_streamer


st.set_page_config(page_title="Real-Time Mask Detector", layout="centered")
st.title("😷 Real-Time Face Mask Detection")
st.write("Click 'Start' and grant camera permissions to detect face masks in real-time.")


@st.cache_resource
def load_mask_model():
    return load_model('mask_detection_MobileNetV2.h5')

model = load_mask_model()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(img_rgb, (224, 224))
    expanded_frame = np.expand_dims(resized_frame, axis=0)
    processed_frame = preprocess_input(expanded_frame)
    
    prediction = model.predict(processed_frame, verbose=0)[0][0]
    
    if prediction > 0.5:
        label = "No Mask"
        color = (0, 0, 255)
    else:
        label = "Mask"
        color = (0, 255, 0)
    
    cv2.putText(img, f"Status: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="mask-detector",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}
)
