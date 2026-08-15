"""
Face Mask Detection — Streamlit App (live webcam via streamlit-webrtc)

Built on top of the MobileNetV2 mask-classifier trained in
`mask_detection_-_MobileNetV2.ipynb`.

Run locally with:
    streamlit run app.py

Deploy on Streamlit Community Cloud (or any HTTPS host) — streamlit-webrtc
needs a secure (https) context for browser camera access, which Streamlit
Cloud provides automatically.

Expects the trained model file `new_mask_detection_MobileNetV2.h5`
to be in the same folder as this script (or upload it via the sidebar).
"""

import os
import threading

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

MODEL_PATH = "mask_detection_MobileNetV2.h5"
IMG_SIZE = 224
LABELS = {0: "Mask", 1: "No Mask"}
COLORS_BGR = {0: (0, 200, 0), 1: (0, 0, 220)}  # green / red
PREDICT_EVERY_N_FRAMES = 5  # skip frames to keep the live feed smooth


# --------------------------------------------------------------------------
# ICE / STUN-TURN configuration
# --------------------------------------------------------------------------
# A public STUN server is enough for local testing, but once this app is
# deployed online, viewers on restrictive/mobile/corporate networks will
# often fail to establish a peer connection without a TURN server (which
# relays traffic when a direct P2P connection isn't possible).
#
# Free options: Twilio's free TURN token API, metered.ca free tier, or
# your own coturn server. Put credentials in Streamlit secrets and they'll
# be picked up automatically below — otherwise it falls back to STUN only.
def get_rtc_configuration() -> RTCConfiguration:
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    turn_url = st.secrets.get("TURN_URL", None) if hasattr(st, "secrets") else None
    turn_username = st.secrets.get("TURN_USERNAME", None) if hasattr(st, "secrets") else None
    turn_credential = st.secrets.get("TURN_CREDENTIAL", None) if hasattr(st, "secrets") else None

    if turn_url and turn_username and turn_credential:
        ice_servers.append(
            {"urls": [turn_url], "username": turn_username, "credential": turn_credential}
        )

    return RTCConfiguration({"iceServers": ice_servers})


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource
def get_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)


def predict_mask(model, frame_bgr: np.ndarray) -> int:
    """Classifies a full frame as Mask (0) / No Mask (1) — mirrors
    detect_face_mask() from the notebook, minus the Haar face crop step."""
    resized = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    reshaped = rgb.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    preprocessed = preprocess_input(np.float32(reshaped))
    prediction = model.predict(preprocessed, verbose=0)
    return int((prediction > 0.5).astype("int32")[0][0])


def draw_label(frame_bgr: np.ndarray, class_id: int) -> np.ndarray:
    label = LABELS[class_id]
    color = COLORS_BGR[class_id]
    annotated = frame_bgr.copy()
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(annotated, (20, 20), (20 + tw + 20, 20 + th + 20), color, cv2.FILLED)
    cv2.putText(annotated, label, (30, 20 + th + 5), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


# --------------------------------------------------------------------------
# streamlit-webrtc video processor
# --------------------------------------------------------------------------
class MaskDetectionProcessor:
    """Runs the classifier on incoming webcam frames. Inference is skipped
    on most frames (only every PREDICT_EVERY_N_FRAMES-th frame is scored)
    and the last known label is reused in between, since a MobileNetV2
    forward pass on every single frame would otherwise stall the video."""

    def __init__(self, model):
        self.model = model
        self.frame_count = 0
        self.last_class_id = 0
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        if self.model is not None:
            self.frame_count += 1
            if self.frame_count % PREDICT_EVERY_N_FRAMES == 0:
                with self.lock:
                    self.last_class_id = predict_mask(self.model, img_bgr)
            img_bgr = draw_label(img_bgr, self.last_class_id)
        else:
            cv2.putText(img_bgr, "No model loaded", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


# --------------------------------------------------------------------------
# Sidebar — model loading
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Model")
uploaded_model = st.sidebar.file_uploader("Upload model (.h5)", type=["h5"])

active_model_path = MODEL_PATH
if uploaded_model is not None:
    active_model_path = "uploaded_model.h5"
    with open(active_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    get_model.clear()  # force reload with the new file

model = get_model(active_model_path)

if model is None:
    st.sidebar.error(
        f"Model file not found at '{MODEL_PATH}'.\n\n"
        "Place `new_mask_detection_MobileNetV2.h5` next to app.py, "
        "or upload it above."
    )
else:
    st.sidebar.success("Model loaded ✅")

# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------
st.title("😷 Face Mask Detector")
st.write(
    "Classifies frames as **Mask** or **No Mask** using a MobileNetV2 model, "
    "either from an uploaded image or a live webcam feed streamed straight "
    "in your browser."
)

mode = st.radio("Choose input source", ["Live Webcam", "Upload Image"], horizontal=True)

if mode == "Live Webcam":
    st.caption(
        "Click **Start** and allow camera access in your browser. "
        "Prediction runs continuously (with light frame-skipping for smoothness)."
    )
    webrtc_streamer(
        key="mask-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=get_rtc_configuration(),
        video_processor_factory=lambda: MaskDetectionProcessor(model),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

else:
    file = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])
    if file is not None:
        if model is None:
            st.warning("Load a model first (see sidebar) before running detection.")
        else:
            image_rgb = np.array(Image.open(file).convert("RGB"))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            with st.spinner("Classifying..."):
                class_id = predict_mask(model, image_bgr)
                annotated_bgr = draw_label(image_bgr, class_id)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption=LABELS[class_id], use_container_width=True)
    else:
        st.info("Upload an image to run mask detection.")

st.divider()
st.caption(
    "Model: MobileNetV2 (frozen base + GlobalAveragePooling2D + Dense sigmoid), "
    "trained for binary mask classification. Live video is streamed via WebRTC "
    "(streamlit-webrtc) directly between your browser and this app — no face "
    "detector is used, the classifier runs on the full frame."
)
