import os
import cv2
import json
import numpy as np
import streamlit as st
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from datetime import datetime
from ultralytics import YOLO
from tamper import store_evidence, verify_chain, get_evidence_count

# -----------------------------
# LOAD ENV + GEMINI
# -----------------------------
load_dotenv()

# N8N webhook URL (optional)
_raw_n8n = os.getenv("N8N_WEBHOOK_URL") or ""
# normalize and strip accidental surrounding whitespace or quotes
N8N_WEBHOOK_URL = _raw_n8n.strip().strip('"').strip("'")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-flash-latest")

# --- N8N helper ---
def send_to_n8n(text, label, confidence, reason):
    payload = {
        "text": text,
        "label": label,
        "confidence": confidence,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print("n8n error:", e)

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
@st.cache_resource
def load_yolo():
    return YOLO("yolo26n_finetuned.pt")

yolo_model = load_yolo()

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(
    page_title="🚨 AI Drug Detection System",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 AI Drug Detection System")
st.caption("Text · Image · Live Webcam · Tamper-Proof Evidence")

tab1, tab2, tab3 = st.tabs(["📝 Text", "🖼 Image", "📷 Webcam"])

# =============================
# 📝 TEXT TAB
# =============================
with tab1:
    text = st.text_area("Enter text to analyze")

    if st.button("Analyze Text") and text.strip():
        prompt = f"""
Classify text into EXACTLY ONE:
Illegal Drug Intent
Drug Promotion
Depression / Mental Health
Safe / Non-Drug

Return JSON only:
{{"label":"","confidence":0.0,"reason":""}}

Text: "{text}"
"""

        try:
            response = gemini_model.generate_content(prompt)
            raw_response = response.text.strip()
            
            # Try to extract JSON from response
            result = None
            try:
                result = json.loads(raw_response)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
            
            if result is None:
                st.error(f"Invalid API response. Raw response: {raw_response}")
                st.stop()

            label = result.get("label", "Unknown")
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "")

            st.success(label)
            st.write("Confidence:", confidence)
            st.write("Reason:", reason)

            # 🔐 Tamper-proof log
            evidence_result = store_evidence({
                "type": "text_analysis",
                "label": label,
                "confidence": confidence,
                "reason": reason,
                "text_length": len(text)
            })
            st.success(f"Evidence stored: Block #{evidence_result['block_index']}")

            # Send text analysis to N8N webhook if configured
            if N8N_WEBHOOK_URL:
                send_to_n8n(text, label, confidence, reason)
                st.info("Sent to N8N")

        except Exception as e:
            st.error(f"Error: {e}")

# =============================
# 🖼 IMAGE TAB
# =============================
with tab2:
    image_file = st.file_uploader("Upload image", ["jpg", "png", "jpeg"])

    if image_file:
        img_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        results = yolo_model(img_bgr)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, caption="YOLO Detection")

        detections = results[0].boxes
        if len(detections) == 0:
            st.info("No objects detected")
        else:
            for box in detections:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]

                st.write(f"{label} — {conf:.2%}")

                # 🔐 Tamper-proof log
                store_evidence({
                    "type": "image_detection",
                    "label": label,
                    "confidence": conf
                })

# =============================
# 📷 WEBCAM TAB
# =============================
with tab3:
    st.warning("Webcam works only in local VS Code")

    start = st.checkbox("Start Webcam")

    if start:
        cap = cv2.VideoCapture(0)
        frame_box = st.empty()
        stats = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame)
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            frame_box.image(annotated_rgb)

            detections = results[0].boxes
            stats.metric("Objects Detected", len(detections))

            for box in detections:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]

                # 🔐 Tamper-proof log
                store_evidence({
                    "type": "webcam_detection",
                    "label": label,
                    "confidence": conf
                })

        cap.release()
