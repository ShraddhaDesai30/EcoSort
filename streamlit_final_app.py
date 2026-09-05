

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
import numpy as np
import cv2
import json
import os
import pandas as pd


# =====================================================
# PATH CONFIGURATION (FIXED)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "plastic_cnn.pth")
META_PATH = os.path.join(BASE_DIR, "meta_classes.json")


# Optional background image (only if you have it)
BG_IMAGE = os.path.join(BASE_DIR, "back3.jpg")

st.set_page_config(page_title="🌿 PrakritiNetraAI", layout="wide")


# =====================================================
# BACKGROUND & THEME (SAFE)
# =====================================================

if os.path.exists(BG_IMAGE):
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("file:///{BG_IMAGE.replace(os.sep, '/')}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1, h2, h3, h4 {{
        color: #1b5e20;
        text-shadow: 1px 1px 2px white;
    }}
    </style>
    """, unsafe_allow_html=True)


# =====================================================
# MODEL SETUP
# =====================================================

# Create meta file if missing
if not os.path.exists(META_PATH):
    meta = {"0": "non-plastic", "1": "plastic"}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

with open(META_PATH, "r") as f:
    meta = json.load(f)

classes = [meta[str(i)] for i in range(len(meta))]

# Load model
weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# APP UI
# =====================================================

st.title("🌿 PrakritiNetraAI – Plastic vs Non-Plastic Detection")
st.write("Detect plastic waste using *CNN + ResNet34 Hybrid Model*")

option = st.radio(
    "Select mode:",
    ["📸 Single Image Detection", "🎥 Real-Time Detection", "🧩 Multi-Object Detection"]
)


# =====================================================
# 1️⃣ SINGLE IMAGE DETECTION
# =====================================================

if option == "📸 Single Image Detection":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t)
            probs = F.softmax(out, dim=1)[0].cpu().numpy()

        idx = np.argmax(probs)
        label = classes[idx]
        conf = probs[idx]

        color = "#ff0000" if label == "plastic" else "#00cc44"
        st.markdown(f"""
        <div style="padding:15px;border-radius:10px;background:black;color:white;
                    text-align:center;border:3px solid {color}">
        <h2>{label.upper()}</h2>
        <h4>Confidence: {conf*100:.2f}%</h4>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# 2️⃣ REAL-TIME DETECTION
# =====================================================

elif option == "🎥 Real-Time Detection":
    cam = st.camera_input("Capture Image")
    if cam:
        img = Image.open(cam).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t)
            probs = F.softmax(out, dim=1)[0].cpu().numpy()

        idx = np.argmax(probs)
        label = classes[idx]
        conf = probs[idx]

        color = "#ff0000" if label == "plastic" else "#00cc44"
        st.markdown(f"""
        <div style="padding:15px;border-radius:10px;background:black;color:white;
                    text-align:center;border:3px solid {color}">
        <h2>{label.upper()}</h2>
        <h4>Confidence: {conf*100:.2f}%</h4>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# 3️⃣ MULTI-OBJECT DETECTION
# =====================================================

elif option == "🧩 Multi-Object Detection":
    choice = st.radio("Select input:", ["📂 Upload Image", "📷 Capture from Camera"])

    img = None
    if choice == "📂 Upload Image":
        file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if file:
            img = Image.open(file).convert("RGB")
    else:
        cam = st.camera_input("Capture Image")
        if cam:
            img = Image.open(cam).convert("RGB")

    if img:
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 40 or h < 40:
                continue

            roi = Image.fromarray(np_img[y:y+h, x:x+w])
            inp = transform(roi).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(inp)
                probs = F.softmax(out, dim=1)[0].cpu().numpy()

            idx = np.argmax(probs)
            label = classes[idx]
            conf = probs[idx]

            color = (0, 255, 0) if label == "non-plastic" else (255, 0, 0)
            cv2.rectangle(np_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(np_img, f"{label} {conf*100:.1f}%",
                        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detected.append({"Label": label, "Confidence (%)": round(conf*100, 2)})

        st.image(np_img, use_container_width=True)
        if detected:
            st.dataframe(pd.DataFrame(detected))
        else:
            st.warning("No objects detected.")
