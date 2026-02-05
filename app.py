import os
import cv2
import json
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
import requests
from dotenv import load_dotenv
from datetime import datetime
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
import feedparser
from tamper import store_evidence, verify_chain, get_evidence_count
import smtplib
import ssl
from email.message import EmailMessage
import base64
import pandas as pd
import time
import re
import PIL.Image
import io

# -----------------------------
# CONFIG & STYLE
# -----------------------------
st.set_page_config(
    page_title="GuardianAI | Intelligent Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

# -----------------------------
# SETUP
# -----------------------------
load_dotenv()

# N8N webhook URL
_raw_n8n = os.getenv("N8N_WEBHOOK_URL") or ""
N8N_WEBHOOK_URL = _raw_n8n.strip().strip('"').strip("'")

# Telegram Params
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_2")
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    client = None

# YOLO
@st.cache_resource
def load_yolo():
    return YOLO("yolo26n_finetuned.pt")

try:
    yolo_model = load_yolo()
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    yolo_model = None

# Slang DB
SLANG_DB_FILE = "slang_db.json"

def load_slang_db():
    if os.path.exists(SLANG_DB_FILE):
        with open(SLANG_DB_FILE, 'r') as f:
            return json.load(f)
    return {"terms": [], "version": "1.0"}

def save_slang_db(db):
    db["last_updated"] = datetime.now().isoformat()
    with open(SLANG_DB_FILE, 'w') as f:
        json.dump(db, f, indent=2)

# Email Configuration
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = "blackivory8494@gmail.com"

# -----------------------------
# SECURITY & ALERTS
# -----------------------------

def get_location_data():
    """Fetch approximate location based on IP"""
    try:
        r = requests.get('http://ip-api.com/json/', timeout=2)
        if r.status_code == 200:
            data = r.json()
            return {
                "lat": float(data.get("lat")), 
                "lon": float(data.get("lon")), 
                "city": data.get("city", "Unknown"),
                "country": data.get("country", "Unknown")
            }
    except:
        pass
    return {"lat": 20.5937, "lon": 78.9629, "city": "Unknown", "country": "India"}

def send_telegram_alert(message, image_bytes=None):
    """Send alert to Telegram Bot"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🚨 GUARDIAN ALERT 🚨\n\n{message}"
    }
    
    try:
        requests.post(url, json=payload, timeout=5)
        
        # Send Photo if available
        if image_bytes:
            photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': image_bytes}
            requests.post(photo_url, data={"chat_id": TELEGRAM_CHAT_ID}, files=files, timeout=10)
            
        return True
    except Exception as e:
        print(f"Telegram Error: {e}")
        return False

def format_alert_msg(reason, location, extra_info=""):
    """Standardized Alert Format"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    city = location.get('city', 'Unknown')
    return f"⏰ Time: {now}\n📍 Location: {city}\n📝 Reason: {reason}\n{extra_info}"

def send_email_warning(label, confidence, reason, source_text):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        return

    msg = EmailMessage()
    msg['Subject'] = f"🚨 DRUG DETECTION ALERT: {label}"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    body = f"""
    ⚠️ HIGH PRIORITY ALERT ⚠️
    
    A potential illegal activity has been detected by GuardianAI.
    
    Type: {label}
    Confidence: {confidence:.2%}
    
    Context/Reason:
    {reason}
    
    Source Input:
    "{source_text}"
    
    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    --
    GuardianAI Automated System
    """
    msg.set_content(body)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        print(f"Failed to send email: {e}")

def send_to_n8n(text, label, confidence, reason):
    if not N8N_WEBHOOK_URL:
        return
    payload = {
        "intent": label,
        "confidence score": confidence,
        "reason": reason
    }
    try:
        requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print("n8n error:", e)

@st.dialog("⚠️ THREAT DETECTED")
def show_threat_alert(label, confidence, reason, text):
    st.markdown(f"<h3 style='color: var(--signal-alert);'>High Priority Alert: {label}</h3>", unsafe_allow_html=True)
    st.write(f"Confidence Level: **{confidence:.1%}**")
    st.markdown("---")
    st.warning(f"**Analysis**: {reason}")
    st.markdown("### Protocol Actions")
    st.checkbox("Notify Command (Email/Telegram)", value=True, disabled=True)
    st.checkbox("Log Evidence to Secure Chain", value=True, disabled=True)
    
    if st.button("Acknowledge & Dismiss", type="primary"):
        st.rerun()

# -----------------------------
# HELPERS
# -----------------------------
def get_dashboard_metrics():
    try:
        with open("evidence_chain.json", "r") as f:
            data = json.load(f)
            chain = data.get("chain", [])
            
        total_scans = len(chain)
        threats = 0
        timestamps = []
        locations = []
        
        for block in chain:
            if block.get("timestamp"):
                timestamps.append(block["timestamp"])
            d = block.get("data", {})
            
            # Map Point
            if "meta" in d and "location" in d["meta"]:
                loc = d["meta"]["location"]
                lat = loc.get("lat")
                lon = loc.get("lon")
                
                if lat is not None and lon is not None:
                    try:
                        locations.append({
                            "lat": float(lat),
                            "lon": float(lon),
                            "type": d.get("label", "Unknown")
                        })
                    except (ValueError, TypeError):
                        pass
            
            if d.get("label") == "Illegal Drug Intent":
                threats += 1
            elif d.get("type") in ["image_detection", "webcam_detection", "vision_analysis", "webcam_vision_threat"] and d.get("confidence", 0) > 0.6:
                 threats += 1
            elif d.get("type") == "webcam_vision_threat": # Always count logic threats
                 threats += 1
                 
        start_time = datetime.fromisoformat(chain[0]["timestamp"]).strftime("%b %d") if chain else "N/A"
        last_scan = datetime.fromisoformat(chain[-1]["timestamp"]).strftime("%H:%M") if chain else "N/A"
        
        # Ensure at least one point for map if empty
        if not locations:
            # Fallback for display
            locations = [{"lat": 20.59, "lon": 78.96, "type": "System Node"}]
            
        return total_scans, threats, start_time, last_scan, chain[-5:], timestamps, locations, data
    except (FileNotFoundError, IndexError, ValueError):
        return 0, 0, "N/A", "N/A", [], [], [], {}

# -----------------------------
# NEW FEATURES
# -----------------------------

def copilot_interface(full_data):
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 AI Copilot")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    user_input = st.sidebar.text_input("Ask about the evidence:", placeholder="Summarize recent threats...", key="copilot_input")
    
    if st.sidebar.button("Ask Copilot", key="ask_btn") and user_input and client:
        with st.sidebar.spinner("Analyzing..."):
            # Limit context
            context_snippet = str(full_data.get("chain", [])[-20:]) 
            prompt = f"""
            You are an AI Investigator Assistant.
            Evidence Log: {context_snippet}
            User Question: {user_input}
            """
            try:
                response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "bot", "content": response.text})
            except Exception as e:
                st.sidebar.error(f"Copilot Error: {e}")

    # Display History
    if st.session_state.chat_history:
        with st.sidebar.expander("Conversation Log", expanded=True):
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "user":
                    st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bot-msg'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

def generate_case_report(metrics_data):
    total, threats, start, last, recent, _, _, _ = metrics_data
    report_content = f"""
    # GUARDIAN AI - INCIDENT REPORT
    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
    
    ## Executive Summary
    - **Total Scans:** {total}
    - **Active Threats:** {threats}
    
    ## Critical Incidents
    """
    for item in recent:
        d = item.get('data', {})
        loc = d.get('meta', {}).get('location', {}).get('city', 'Unknown')
        if d.get('confidence', 0) > 0.5 or d.get('type') == 'webcam_vision_threat':
             report_content += f"- **{d.get('type').upper()}** in {loc}: {d.get('description', d.get('label', 'N/A'))}\n"
    return report_content

# -----------------------------
# PAGES
# -----------------------------

def render_kpi_row(metrics):
    """Key Performance Indicators Row"""
    total, threats, _, _, _, _, _, _ = metrics
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Inspections", total)
    with c2: st.metric("Threat Incidents", threats, delta="Action Required" if threats > 0 else "Normal")
    with c3: st.metric("System Uptime", "99.98%", delta="+0.01%")
    with c4: st.metric("Active Agents", "3 Online", delta="Gemini 2.5")

def render_log_list(recent_data):
    """Renders a vertical list of recent logs"""
    if not recent_data:
        st.info("No activity recorded.")
        return
    
    # Process newest first
    for item in reversed(recent_data):
        d = item.get("data", {})
        label = d.get("label") or d.get("description") or "Unknown"
        # Location info
        loc = d.get("meta", {}).get("location", {})
        city = loc.get("city", "Unknown")
        
        # Clean formatting
        dtype = d.get("type", "General").replace("_", " ").title()
        time_str = item.get("timestamp", "").split("T")[1][:8]
        
        # Color code
        is_threat = d.get("confidence", 0) > 0.6 or "illegal" in label.lower() or "threat" in label.lower()
        border_col = "var(--neon-red)" if is_threat else "var(--glass-border)"
        
        st.markdown(f"""
        <div style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid var(--border-color);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="font-weight:600; color:var(--text-primary);">{dtype}</span>
                <span style="font-size:0.8rem; color:var(--text-secondary);">{time_str}</span>
            </div>
            <div style="color:var(--text-secondary); font-size:0.9rem;">{label}</div>
            <div style="font-size:0.75rem; color:var(--brand-secondary); margin-top:4px; font-weight:500;">
                 📍 {city.upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_geo_trace(full_data):
    """Renders Location Intelligence Card"""
    st.subheader("Location Intelligence")
    
    # Get latest item with location
    chain = full_data.get("chain", [])
    target = None
    if chain:
        target = chain[-1] 
    
    if target:
        d = target.get("data", {})
        loc = d.get("meta", {}).get("location", {})
        lat = loc.get("lat")
        lon = loc.get("lon")
        city = loc.get("city", "Unknown Sector")
        
        if lat and lon:
            # Display Coordinates
            st.markdown(f"""
            <div style="margin-bottom:15px;">
                <div style="font-size:0.875rem; color:var(--text-secondary);">Latest Detection Event</div>
                <div style="font-size:1.25rem; font-weight:600; color:var(--text-primary);">{city}</div>
                <div style="font-family:monospace; font-size:0.8rem; color:var(--brand-primary);">{lat:.4f}, {lon:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Focused Map
            try:
                df = pd.DataFrame([{"lat": float(lat), "lon": float(lon)}])
                st.map(df, zoom=10, height=300)
            except Exception:
                st.error("Map Data Unavailable")
        else:
            st.info("No geospatial data in latest event.")
    else:
        st.info("Awaiting data stream...")

def render_dashboard():
    # Hero Section - Clean & Professional
    st.title("Executive Dashboard")
    st.caption("Real-time narcotics interdiction monitoring and forensics.")
    
    metrics_raw = get_dashboard_metrics()
    total, threats, started, last_active, recent, timestamps, locations, full_data = metrics_raw
    
    # 1. KPI Row
    render_kpi_row(metrics_raw)
    st.markdown("---")
    
    # 2. Main Grid
    # 2. Main Grid
    # 2. Main Grid
    col_left, col_mid, col_right = st.columns([2, 1, 1])
    
    with col_left:
        with st.container(height=500, border=True):
            st.subheader("Geospatial Threat Distribution")
            map_df = pd.DataFrame(locations)
            st.map(map_df, zoom=2, use_container_width=True)
        
    with col_mid:
        with st.container(height=500, border=True):
            render_geo_trace(full_data)

    with col_right:
        with st.container(height=500, border=True):
            st.subheader("Incident Activity Log")
            render_log_list(recent)

    # 3. Evidence Timeline (Horizontal)
    st.markdown("### 🛑 Recent Interceptions")
    if recent:
        cols = st.columns(len(recent))
        for idx, item in enumerate(recent):
            d = item.get("data", {})
            with cols[idx]:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <div style="font-size:0.8rem; color:var(--neon-cyan)">{item.get('timestamp', '')[11:16]}</div>
                    <div style="font-weight:bold; margin: 5px 0;">{d.get('type', 'Unknown').replace('_', ' ').title()}</div>
                    <div style="font-size:0.8rem; color:#888;">{d.get('label', 'Alert')}</div>
                </div>
                """, unsafe_allow_html=True)

    # Copilot Integration
    copilot_interface(full_data)
    
    st.markdown('<div class="scanner-overlay"><div class="scanner-line"></div></div>', unsafe_allow_html=True)


def render_text_analysis():
    st.title("📝 Semantic Intelligence (Adversarial Defense)")
    st.caption("Advanced detection for Leet speak, Emojis, Mixed Languages, and Slang.")
    
    slang_db = load_slang_db()
    known_slang = ", ".join(slang_db.get("terms", []))
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        text_input = st.text_area("Comms Intercept", height=150, placeholder="Paste suspect text (e.g. 'Got any sn0w? ❄️')...")
        
        if st.button("RUN DEEP SCAN", type="primary"):
            if not text_input.strip(): return
            
            with st.spinner("Deciphering evasion techniques..."):
                # Adversarial Prompt
                prompt = f"""
                Analyze the following text for illegal drug intent.
                
                You must Detect:
                1. Leet speak (e.g. "w33d", "c0caine")
                2. Emoji Steganography (e.g. 🍁, ❄️, 🍄, 💊 used for drugs)
                3. Mixed Languages (Hinglish, Spanglish)
                4. Street Slang (Known local terms: {known_slang})
                
                Classify into EXACTLY ONE: 'Illegal Drug Intent', 'Drug Promotion', 'Depression/Health', 'Safe'.
                
                Return JSON only: {{"label":"","confidence":0.0,"reason":"","detected_slang_term": ""}}
                
                Text: "{text_input}"
                """
                if not client:
                    st.error("AI Engine Offline")
                    return
                    
                resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                
                try:
                    match = re.search(r'\{.*\}', resp.text, re.DOTALL)
                    res = json.loads(match.group(0)) if match else {"label": "Error", "confidence": 0}
                    
                    label = res.get("label")
                    conf = res.get("confidence")
                    reason = res.get("reason")
                    slang_term = res.get("detected_slang_term", "")
                    
                    # Store Result with Geo
                    current_loc = get_location_data()
                    store_evidence({
                        "type": "text_analysis", 
                        "label": label, 
                        "confidence": conf, 
                        "reason": reason,
                        "adversarial_check": True,
                        "meta": {"location": current_loc}
                    })
                    
                    # Display
                    color = "var(--signal-alert)" if "Illegal" in label else "var(--signal-safe)"
                    st.markdown(f"""
                    <div style="padding: 20px; border: 1px solid {color}; border-radius: 10px; background: rgba(0,0,0,0.2);">
                        <h2 style="color: {color}; margin:0;">{label.upper()}</h2>
                        <h1 style="font-size: 3.5rem; color: #fff;">{conf:.1%}</h1>
                        <p style="color: #ccc;">{reason}</p>
                        <p style="font-size: 0.8rem; color: #888;">Trace: {current_loc.get('city', 'Unknown')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Alerts
                    if "Illegal" in label and conf > 0.7:
                        # Send Telegram
                        if TELEGRAM_BOT_TOKEN:
                            msg = format_alert_msg(reason, current_loc, f"Input: {text_input}")
                            send_telegram_alert(msg)
                        
                        show_threat_alert(label, conf, reason, text_input)

                    # Adaptive Learning Opportunity
                    if "Illegal" in label and slang_term and slang_term.lower() not in slang_db["terms"]:
                        st.info(f"New potential slang detected: **{slang_term}**")
                        if st.button(f"Add '{slang_term}' to Database"):
                            slang_db["terms"].append(slang_term.lower())
                            save_slang_db(slang_db)
                            st.success(f"Database updated. System will now recognize '{slang_term}'.")
                            
                except Exception as e:
                    st.error(f"Parsing Error: {e}")

    with c2:
        st.markdown("#### Database Status")
        st.write(f"Learning Terms: **{len(slang_db['terms'])}**")
        with st.expander("View Dictionary"):
            st.write(slang_db['terms'])

def render_image_analysis():
    st.title("👁️ Visual Forensics & OCR")
    st.caption("Combined Object Detection (YOLO) and Semantic Image Analysis (Gemini Vision).")

    uploaded_file = st.file_uploader("Upload Imagery", type=['jpg','png','jpeg'])
    
    if uploaded_file:
        img_bytes = uploaded_file.getvalue()
        
        if yolo_model:
            img_arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
            img_bgr = cv2.imdecode(img_arr, 1)
            
            c1, c2 = st.columns(2)
            with c1: st.image(img_bytes, caption="Source", use_container_width=True)
            
            if st.button("RUN ADVANCED SCAN", type="primary"):
                with c2:
                    with st.spinner("Multi-modal analysis..."):
                        # YOLO
                        res = yolo_model(img_bgr)
                        plot = res[0].plot()
                        st.image(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB), caption="Object Layer", use_container_width=True)
                        
                        obj_count = len(res[0].boxes)
                        
                        # Gemini Vision
                        st.write("Generating Semantic Analysis...")
                        try:
                            prompt = "Analyze this image efficiently. Return valid JSON only: {'is_threat': bool, 'description': str}. Is it containing illegal drugs, weapons, or TEXT related to selling drugs?"
                            pil_img = PIL.Image.open(io.BytesIO(img_bytes))
                            
                            if client:
                                g_resp = client.models.generate_content(
                                    model="gemini-2.5-flash",
                                    contents=[prompt, pil_img]
                                )
                                gemini_raw = g_resp.text
                                
                                # Clean JSON Extraction
                                match = re.search(r'\{.*\}', gemini_raw, re.DOTALL)
                                if match:
                                    vision_data = json.loads(match.group(0))
                                else:
                                    vision_data = {"is_threat": False, "description": "Analysis format error.", "raw": gemini_raw}
                                
                            else:
                                vision_data = {"is_threat": False, "description": "Gemini Offline"}

                        except Exception as e:
                            vision_data = {"is_threat": False, "description": f"Vision Error: {e}"}
                        
                        # Logic
                        gemini_desc = vision_data.get("description", "No Data")
                        is_threat_ai = vision_data.get("is_threat", False)
                        
                        is_threat = obj_count > 0 or is_threat_ai
                        
                        if is_threat:
                            st.error("⚠️ THREAT CONFIRMED")
                            st.write(f"**Analysis:** {gemini_desc}")
                            
                            current_loc = get_location_data()
                            store_evidence({
                                "type": "vision_analysis", 
                                "yolo_count": obj_count, 
                                "description": gemini_desc,
                                "meta": {"location": current_loc}
                            })
                            
                            if TELEGRAM_BOT_TOKEN:
                                msg = format_alert_msg(gemini_desc, current_loc)
                                send_telegram_alert(msg, image_bytes=img_bytes)
                        else:
                            st.success("✅ Analysis Clear")
                            st.info(f"**Insight:** {gemini_desc}")

def render_webcam():
    st.title("🔴 Live Surveillance")
    
    # Session State for Narrative
    if "narrative_events" not in st.session_state:
        st.session_state.narrative_events = []
    
    c1, c2 = st.columns([3, 1])
    active = st.toggle("ACTIVATE CAMERA FEED", value=False)
    enable_ai_vision = st.toggle("🤖 AI AUTO-VERIFY (Gemini)", value=False)
    
    # Narrative Controls
    with c2:
        st.metric("Camera Status", "LIVE" if active else "OFFLINE")
        st.caption(f"Node: {get_location_data().get('city', 'Unknown')}")
        log_ph = st.empty()
        vision_ph = st.empty()
        
        st.markdown("---")
        st.subheader("📜 Narrative Engine")
        if st.button("Generate Timeline", help="Reconstruct events into a story"):
            if not st.session_state.narrative_events:
                st.info("No recent events to reconstruct.")
            else:
                with st.spinner("Reconstructing timeline..."):
                    # Prompt Gemini for Narrative
                    event_log = str(st.session_state.narrative_events[-15:]) # Analyze last 15 events
                    prompt = f"""
                    You are a Forensic Narrative Expert.
                    Convert the following raw detection logs into a chronological detection narrative.
                    
                    Input Logs: {event_log}
                    
                    Rules:
                    1. Format as a compelling, professional timeline story.
                    2. Example start: "At [Time], an object consistent with [Object] appeared..."
                    3. Mention confidence trends (e.g., "Confidence increased to 95%...").
                    4. Keep it concise but detailed.
                    
                    Output:
                    """
                    if client:
                        try:
                            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                            st.session_state.narrative_result = resp.text
                        except Exception as e:
                            st.error(f"Narrative Error: {e}")
                            
        if "narrative_result" in st.session_state:
            st.markdown(f"<div style='font-family: monospace; font-size: 0.9em; background: #222; padding: 10px; border-radius: 5px; border-left: 3px solid cyan;'>{st.session_state.narrative_result}</div>", unsafe_allow_html=True)

    with c1:
        frame_ph = st.empty()
        
    if active and yolo_model:
        cap = cv2.VideoCapture(0)
        
        last_analysis_time = 0
        
        while active:
            ret, frame = cap.read()
            if not ret: break
            
            res = yolo_model(frame)
            annotated = res[0].plot()
            frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # YOLO Logging
            if len(res[0].boxes) > 0:
                box = res[0].boxes[0]
                cls = yolo_model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                now_str = datetime.now().strftime("%H:%M:%S")
                
                # Update visual log
                log_ph.warning(f"DETECTED: {cls} ({conf:.0%})")
                
                # Append to Narrative State
                st.session_state.narrative_events.append({
                    "time": now_str,
                    "event": "Object Detection",
                    "object": cls,
                    "confidence": f"{conf:.2f}"
                })
                
                # Standard Evidence Log Logic (Random Sampling)
                if np.random.rand() > 0.95: 
                    current_loc = get_location_data()
                    store_evidence({
                        "type": "webcam_detection", 
                        "label": cls, 
                        "confidence": conf,
                        "meta": {"location": current_loc}
                    })
            
            # Gemini Vision Analysis (Rate Limited)
            if enable_ai_vision and (time.time() - last_analysis_time > 5):
                last_analysis_time = time.time()
                vision_ph.info("🔄 Scanning...")
                
                try:
                    # Convert Frame to Bytes
                    _, buf = cv2.imencode('.jpg', frame)
                    img_bytes = buf.tobytes()
                    pil_img = PIL.Image.open(io.BytesIO(img_bytes))
                    
                    if client:
                        prompt = "Analyze this surveillance frame efficiently. Is there any illegal drug activity, weapons, or suspicious behavior? Return JSON only: {'is_threat': bool, 'description': str}"
                        
                        g_resp = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[prompt, pil_img]
                        )
                        gemini_raw = g_resp.text
                        
                        # Clean JSON Extraction
                        match = re.search(r'\{.*\}', gemini_raw, re.DOTALL)
                        if match:
                            vision_data = json.loads(match.group(0))
                        else:
                            vision_data = {"is_threat": False, "description": "Analysis format error."}
                            
                        # Process Result
                        is_threat = vision_data.get("is_threat", False)
                        desc = vision_data.get("description", "No details.")
                        now_str = datetime.now().strftime("%H:%M:%S")
                        
                        # Append to Narrative State (only if Logic Vision runs)
                        if is_threat:
                             st.session_state.narrative_events.append({
                                "time": now_str,
                                "event": "Vision Threat",
                                "description": desc
                            })
                        
                        if is_threat:
                            vision_ph.error(f"🚨 {desc}")
                            current_loc = get_location_data()
                            
                            # Log Threat
                            store_evidence({
                                "type": "webcam_vision_threat",
                                "description": desc,
                                "meta": {"location": current_loc}
                            })
                            
                            # Alert
                            if TELEGRAM_BOT_TOKEN:
                                msg = format_alert_msg(f"Live Vision Threat: {desc}", current_loc)
                                send_telegram_alert(msg, image_bytes=img_bytes)
                        else:
                            vision_ph.success(f"Clear: {desc}")
                            
                    else:
                        vision_ph.warning("Gemini Offline")
                        
                except Exception as e:
                    vision_ph.error(f"Vision Error: {e}")

        cap.release()


def render_audio_analysis():
    st.title("🎙️ Audio Forensics")
    st.caption("Voice-based threat detection using Gemini 1.5 Flash (Multimodal).")

    # Input Methods
    st.subheader("Input Audio")
    
    # Tab layout for input types
    tab1, tab2 = st.tabs(["🔴 Live Recording", "Tk Upload Audio File"])
    
    audio_bytes = None
    mime_type = "audio/wav" # Default for streamlit recorder
    
    with tab1:
        # Streamlit Audio Recorder
        # Note: st.audio_input is available in newer streamlit versions
        try:
            audio_value = st.audio_input("Record Voice Evidence")
            if audio_value:
                audio_bytes = audio_value.getvalue()
                st.audio(audio_bytes, format="audio/wav")
        except AttributeError:
             st.warning("Your Streamlit version might be too old for st.audio_input. Please use file upload.")

    with tab2:
        uploaded_audio = st.file_uploader("Upload Audio File", type=['wav', 'mp3', 'm4a', 'ogg'])
        if uploaded_audio:
            audio_bytes = uploaded_audio.getvalue()
            mime_type = uploaded_audio.type
            st.audio(audio_bytes, format=mime_type)

    if audio_bytes:
        if st.button("RUN AUDIO ANALYSIS", type="primary"):
            with st.spinner("Listening & Analyzing..."):
                if not client:
                    st.error("AI Engine Offline")
                    return

                try:
                    # Construct Prompt
                    prompt = """
                    Analyze this audio recording for illegal drug-related activities, drug dealing, or immediate threats.
                    
                    Return JSON only:
                    {
                        "is_threat": boolean,
                        "confidence": float (0.0 to 1.0),
                        "transcription": "Verbatim transcript of the audio",
                        "summary": "Brief summary of the context",
                        "threat_details": "Why is this a threat?"
                    }
                    
                    Strictly ignore harmless conversations. Only flag clear contexts of drug trade or violence.
                    """
                    
                    # Prepare content for Gemini
                    # Using types.Part to pass raw bytes
                    
                    # Mapping common mimes to what Gemini accepts if needed, 
                    # generally it accepts standard audio/* types.
                    
                    # Note: The google.genai V1 SDK usage:
                    # contents=[prompt, types.Part.from_bytes(data, mime_type)]
                    
                    try:
                        # Attempt to use the client directly with parts
                        # We need to ensure we are using the correct Part structure for the specific SDK version imported.
                        # Based on typical usage:
                        
                        from google.genai import types
                        
                        part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
                        
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[prompt, part]
                        )
                        
                        # Process Response
                        raw_text = response.text
                        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                        
                        if match:
                            result = json.loads(match.group(0))
                        else:
                            result = {"is_threat": False, "transcription": "Error parsing JSON", "summary": raw_text}
                            
                        # Display Results
                        is_threat = result.get("is_threat", False)
                        conf = result.get("confidence", 0.0)
                        transcript = result.get("transcription", "N/A")
                        details = result.get("threat_details", "None")
                        
                        st.markdown("### Analysis Report")
                        c1, c2 = st.columns([2, 1])
                        
                        with c1:
                            st.markdown(f"**Transcript:**\n*{transcript}*")
                            if is_threat:
                                st.error(f"🚨 THREAT DETECTED (Confidence: {conf:.0%})")
                                st.write(f"**Reason:** {details}")
                            else:
                                st.success("✅ Content Safe")
                                st.write(f"**Summary:** {result.get('summary')}")
                                
                        with c2:
                            st.metric("Threat Score", f"{conf:.0%}", delta="High Risk" if is_threat else "Low Risk", delta_color="inverse")
                            
                        # Actions
                        if is_threat:
                            current_loc = get_location_data()
                            # Store Evidence
                            store_evidence({
                                "type": "audio_analysis",
                                "label": "Voice Threat",
                                "confidence": conf,
                                "transcription": transcript,
                                "reason": details,
                                "meta": {"location": current_loc}
                            })
                            
                            # Alert
                            if TELEGRAM_BOT_TOKEN:
                                msg = format_alert_msg(details, current_loc, f"Transcript: {transcript}")
                                send_telegram_alert(msg)
                                
                            show_threat_alert("Voice Threat Detected", conf, details, transcript)
                            
                    except Exception as inner_e:
                        st.error(f"Gemini API Error: {inner_e}")
                        
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")


# -----------------------------
# MAIN NAVIGATION
# -----------------------------

with st.sidebar:
    st.title("🛡️ GUARDIAN")
    menu = st.radio("MODULES", ["Dashboard", "Semantic Intelligence", "Visual Forensics", "Audio Intelligence", "Live Surveillance"])
    st.markdown("---")

if menu == "Dashboard":
    render_dashboard()
elif menu == "Semantic Intelligence":
    render_text_analysis()
elif menu == "Visual Forensics":
    render_image_analysis()
elif menu == "Audio Intelligence":
    render_audio_analysis()
elif menu == "Live Surveillance":
    render_webcam()

