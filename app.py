import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av


st.set_page_config(page_title="Face Mask Detector", layout="centered")
st.title("Real-Time Face Mask Detector")
st.write("Click 'Start' to activate your webcam. The model will detect if you are wearing a mask.")


@st.cache_resource
def load_mask_model():
    return load_model("mask_detection_MobileNetV2.h5")

@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

model = load_mask_model()
face_cascade = load_cascade()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        face_crop = img[y:y+h, x:x+w]
        
        resized_face = cv2.resize(face_crop, (224, 224))
        
        reshaped_face = np.reshape(resized_face, (1, 224, 224, 3))
        
        normalized_face = reshaped_face / 255.0
        
        prediction = model.predict(normalized_face)
        
        if prediction[0][0] < 0.5:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)
            
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="mask-detector",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
