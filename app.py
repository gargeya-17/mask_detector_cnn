import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Real-Time Mask Detector",layout="centered")
st.title("😷 Real-Time Face Mask Detection")
st.write("Click 'Start' to open your webcam and detect face masks in real-time.")


@st.cache_resource
def load_mask_model():
    return load_model('mask_detection_MobileNetV2.h5')

model=load_mask_model()

run_camera=st.checkbox("Start Webcam")
FRAME_WINDOW=st.image([])

if run_camera:
    cap=cv2.VideoCapture(0)
    
    while run_camera:
        ret,frame=cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break
            
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        resized_frame=cv2.resize(frame_rgb,(224,224)) 
        expanded_frame=np.expand_dims(resized_frame,axis=0)
        processed_frame=preprocess_input(expanded_frame)
        
        prediction=model.predict(processed_frame,verbose=0)[0][0]
        
        if prediction > 0.5:
            label="No Mask"
            color=(0,0,255)
        else:
            label="Mask"
            color=(0,255,0)
        
        cv2.putText(frame, f"Status: {label}",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)
        
        frame_render=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_render)
        
    cap.release()
else:
    st.write("Webcam is turned off.")