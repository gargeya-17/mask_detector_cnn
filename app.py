import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 1. Setup Streamlit Page ---
st.set_page_config(page_title="Mask Detector", layout="centered")
st.title("📸 Snapshot Face Mask Detection")
st.write("Click the button below to take a photo, and the AI will check if you are wearing a mask.")

# --- 2. Load the trained model ---
@st.cache_resource
def load_mask_model():
    # Make sure this matches your actual model filename
    return load_model('mask_detection_MobileNetV2.h5')

model = load_mask_model()

# --- 3. Camera Input Widget ---
# This single line handles opening the webcam and taking the photo
camera_photo = st.camera_input("Take a picture")

# --- 4. Process the Photo ---
if camera_photo is not None:
    # Read the image file as a PIL Image and ensure it is in RGB format
    img = Image.open(camera_photo).convert('RGB')
    img_array = np.array(img)
    
    # Preprocess the image exactly as done during training
    resized_img = cv2.resize(img_array, (224, 224))        # Resize to 224x224
    expanded_img = np.expand_dims(resized_img, axis=0)     # Expand dims for batch size
    processed_img = preprocess_input(expanded_img)         # Apply MobileNetV2 preprocessing
    
    # Make Prediction
    with st.spinner("Analyzing photo..."):
        prediction = model.predict(processed_img, verbose=0)[0][0]
    
    # Display the result
    # 0 -> 'with_mask', 1 -> 'without_mask'
    st.divider()
    if prediction > 0.5:
        st.error("🚨 **Result: No Mask Detected**")
    else:
        st.success("✅ **Result: Mask Detected**")
