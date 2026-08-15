import streamlit as st
import cv2
import numpy as np
import av
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_webrtc import webrtc_streamer

# --- 1. Setup Streamlit Page ---
st.set_page_config(page_title="Real-Time Mask Detector", layout="centered")
st.title("😷 Real-Time Face Mask Detection")
st.write("Click 'Start' and grant camera permissions to detect face masks in real-time.")

# --- 2. Load and Warm-Up the Model ---
@st.cache_resource
def load_and_warmup_model():
    model = load_model('mask_detection_MobileNetV2.h5')
    
    # WARM-UP: Run a dummy prediction in the main thread to initialize the TF graph.
    # This completely prevents the Segmentation Fault in the background thread.
    dummy_frame = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model(dummy_frame, training=False)
    
    return model

model = load_and_warmup_model()

# --- 3. WebRTC Video Processing ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert the WebRTC frame to an OpenCV numpy array (BGR format)
    img = frame.to_ndarray(format="bgr24")
    
    # Preprocess the entire frame exactly as done during training
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # Convert BGR to RGB
    resized_frame = cv2.resize(img_rgb, (224, 224))          # Resize to 224x224
    expanded_frame = np.expand_dims(resized_frame, axis=0)   # Expand dims for batch size
    processed_frame = preprocess_input(expanded_frame)       # Apply MobileNetV2 preprocessing
    
    # Make Prediction using the pre-warmed, thread-safe model call
    prediction_tensor = model(processed_frame, training=False)
    prediction = prediction_tensor.numpy()[0][0]
    
    # 0 -> 'with_mask', 1 -> 'without_mask'
    if prediction > 0.5:
        label = "No Mask"
        color = (0, 0, 255) # Red for no mask
    else:
        label = "Mask"
        color = (0, 255, 0) # Green for mask
        
    # Draw the overall prediction label on the top-left corner of the frame
    cv2.putText(img, f"Status: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Return the processed frame to be displayed in the browser
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. Streamlit WebRTC Component ---
webrtc_streamer(
    key="mask-detector",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}
)
