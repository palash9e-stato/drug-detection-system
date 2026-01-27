import os
import cv2
import json
import requests
import streamlit as st
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from ultralytics import YOLO

# ----------------
# LOAD SECRETS

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-flash-latest")

# -------------------
# LOAD YOLO MODEL

@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

yolo_model = load_yolo_model()

# ------------------
# STREAMLIT CONFIG

st.set_page_config(
    page_title=" AI Drug Detection System",
    page_icon="🚨",
    layout="centered"
)

st.title(" AI Drug Detection System")
st.caption("Text · Image · Live Webcam · Alerts")

tab1, tab2, tab3 = st.tabs([" TEXT", " IMAGE", " WEBCAM"])

# --------------------------------------------------
# SEND TO N8N
# --------------------------------------------------
def send_to_n8n(payload):
    if not N8N_WEBHOOK_URL:
        return {"ok": False, "error": "N8N_WEBHOOK_URL not set"}
    try:
        resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
        return {"ok": resp.ok, "status_code": resp.status_code, "text": resp.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ----------
# TEXT TAB

with tab1:
    text = st.text_area("Enter text")

    if st.button("Analyze Text"):
        prompt = f"""
Classify text into ONE category:
Illegal Drug Intent
Drug Promotion
Depression / Mental Health
Safe / Non-Drug

Return JSON:
{{"label":"","confidence":0.0,"reason":""}}

Text: "{text}"
"""

        response = model.generate_content(prompt)
        raw_text = getattr(response, "text", str(response))

        # Try to parse JSON strictly, then try to extract a JSON substring, otherwise keep raw
        result = None
        try:
            result = json.loads(raw_text)
        except Exception:
            # attempt to extract first JSON object found in the text
            import re

            m = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group(0))
                except Exception:
                    result = None

        if result is None:
            st.error("Model response could not be parsed as JSON — showing raw response for debugging.")
            with st.expander("Raw model response"):
                st.text(raw_text)
            result = {"label": "Unknown", "confidence": 0.0, "reason": raw_text}
        else:
            # show parsed result for visibility
            st.subheader("Parsed model result")
            st.json(result)

        # Display primary fields
        label = result.get("label") or result.get("Label") or result.get("prediction") or "Unknown"
        confidence = result.get("confidence") or result.get("score") or 0.0
        reason = result.get("reason") or result.get("explanation") or ""

        st.success(label)
        st.write("Confidence:", confidence)
        st.write("Reason:", reason)

        # Send to n8n and report status
        n8n_response = send_to_n8n({
            "type": "text",
            **result,
            "timestamp": datetime.utcnow().isoformat()
        })
        if not n8n_response.get("ok"):
            st.warning(f"N8N webhook not sent: {n8n_response.get('error') or n8n_response.get('text')}" )
        else:
            st.info(f"Sent to N8N (status {n8n_response.get('status_code')})")

    # Manual N8N test button
    if st.button("Test N8N Webhook"):
        test_payload = {
            "type": "test",
            "message": "manual webhook test",
            "timestamp": datetime.utcnow().isoformat()
        }
        resp = send_to_n8n(test_payload)
        if resp.get("ok"):
            st.success(f"Webhook sent successfully (status {resp.get('status_code')})")
            st.json(resp)
        else:
            st.error(f"Webhook failed: {resp.get('error') or resp.get('text')}")
            st.json(resp)

# ----------
# IMAGE TAB 

with tab2:
    img = st.file_uploader("Upload image", ["jpg", "png"])
    if img:
        # Convert uploaded file to image
        img_array = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        results = yolo_model(img_bgr)
        
        # Draw detections on image
        detected_img = results[0].plot()
        detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
        
        # Display original and detected images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(img_rgb)
        
        with col2:
            st.subheader("YOLO Detection")
            st.image(detected_img_rgb)
        
        # Show detection results
        st.subheader("Detection Results")
        detections = results[0].boxes
        if len(detections) > 0:
            st.success(f"Found {len(detections)} object(s)")
            for i, box in enumerate(detections):
                conf = box.conf[0].item()
                st.write(f"Object {i+1} - Confidence: {conf:.2%}")
        else:
            st.info("No objects detected")

# ------------
# WEBCAM TAB 

with tab3:
    st.warning("Webcam works only in local VS Code")

    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        frame_holder = st.empty()
        stats_holder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = yolo_model(frame)
            
            # Draw detections on frame
            detected_frame = results[0].plot()
            detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            frame_holder.image(detected_frame_rgb)
            
            
            detections = results[0].boxes
            stats_holder.metric("Objects Detected", len(detections))

        cap.release()
