# GuardianAI — Intelligent Drug Detection & Surveillance Platform

GuardianAI is a powerful, AI-driven surveillance and narcotics detection app built with Python and Streamlit. It combines computer vision, natural language processing, audio analysis, and blockchain-style evidence storage — all inside a single web-based dashboard. Whether you're learning about AI security systems or building on top of this project, this guide walks you through everything.

---

## Table of Contents

1. [What Does This App Do?](#what-does-this-app-do)
2. [Features Overview](#features-overview)
3. [How It Works — Module by Module](#how-it-works--module-by-module)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Setup & Installation](#setup--installation)
7. [Environment Variables](#environment-variables)
8. [Running the App](#running-the-app)
9. [Security Features](#security-features)


---

## What Does This App Do?

GuardianAI is designed to help detect illegal drug-related activity using multiple types of inputs — images, videos, live webcam feeds, text messages, and audio recordings. It uses AI models (YOLO for object detection, Google Gemini for multimodal understanding) to analyze content and flag threats. All detections are logged into a tamper-proof, blockchain-inspired evidence chain that cannot be edited without detection.

Think of it as a digital surveillance officer that can see, hear, read, and remember — and alert a human operator when something suspicious is found.

---

## Features Overview

| Module | What It Does |
|--------|-------------|
| **Dashboard** | Command center showing all recent detections, locations, and threat stats |
| **Visual Forensics** | Upload images — YOLO + Gemini Vision analyze them for drugs or contraband |
| **Live Surveillance** | Real-time webcam feed with AI object detection, behavioral analysis, and auto-alerts |
| **Semantic Intelligence** | Analyze text messages for coded language, slang, emojis, and leet speak |
| **Audio Intelligence** | Analyze voice recordings for drug-related content; includes voice ID matching |
| **Video Forensics** | Upload video files for frame-by-frame YOLO scanning |
| **3D Scene Reconstruction** | Convert a 2D image into a navigable 3D point cloud |
| **Criminal Network Graph** | Visualize connections between suspects, locations, and detected objects |
| **Whistleblower Portal** | Anonymously submit evidence using Zero-Knowledge Proofs |
| **Evidence Chain** | Every detection is stored in a cryptographic blockchain-like log |
| **Telegram Alerts** | Automatic alerts with images sent to a Telegram bot on threat detection |
| **AI Copilot** | Sidebar chatbot powered by Gemini to answer questions about the evidence log |
| **Ghost / Panic Mode** | Instantly disguise the app as an office inventory system |
| **Dead Man's Switch** | Background security daemon that locks the app if tampered with |

---

## How It Works — Module by Module

### 1. Dashboard

The first screen you see is the Executive Dashboard. It shows:

- **Total Inspections** — how many scans have been recorded
- **Threat Incidents** — how many flagged events exist
- **System Uptime** — always-on status indicator
- **Swarm Link** — how many networked nodes are connected
- **3D Geospatial Threat Map** — all detections plotted on an interactive 3D hexagon heatmap using PyDeck
- **Pre-Crime Forecast** — click "Generate Heatmap" and the app uses past detection locations to predict future hot zones
- **Recent Interceptions** — a timeline of the last 5 detections
- **Incident Activity Log** — detailed cards showing each event, ZK-Proof badge, and blockchain status

---

### 2. Visual Forensics (Image Analysis)

Upload a `.jpg`, `.png`, or `.jpeg` image.

The app runs **two parallel analyses**:

- **YOLO Object Detection** — a custom-trained YOLO model (`yolo26n_finetuned.pt`) draws bounding boxes around any detected suspicious items
- **Gemini Vision** — Google's Gemini 2.5 Flash model reads the image semantically and returns whether it's a threat, with a short description

If either model flags a threat:
- A red alert banner appears
- The detection is logged to the evidence chain (with your current GPS location)
- A Telegram alert is sent with the image attached

---

### 3. Live Surveillance (Webcam)

Connect a webcam (local device or IP camera URL) and GuardianAI processes every frame in real time.

Features active during live feed:

- **YOLO Detection** — bounding boxes drawn on screen for every detected object
- **AI Auto-Verify** — every 5 seconds, Gemini Vision checks the current frame for suspicious activity
- **Privacy Blur (Eyes-On-Glass)** — if the operator looks away or no eyes are detected, the feed blurs automatically. This prevents unattended surveillance
- **Spatial Tripwire** — draw an invisible horizontal line on screen. Any movement crossing it triggers an alarm
- **Behavioral Profiling (Psycho-Pass)** — analyzes body pose via the MediaPipe Pose Landmarker to estimate stress level, detect surrender postures, and flag aggressive behavior
- **Auto-Lock** — if no person is detected for 30 seconds, the console locks itself and sends a Telegram alert
- **Narrative Engine** — click "Generate Timeline" and Gemini reconstructs all the recent detections into a written chronological story
- **IP Camera Support** — paste any MJPEG stream URL (e.g., from your phone's IP Webcam app) to use it as the feed source

---

### 4. Semantic Intelligence (Text Analysis)

Paste any suspicious text message or communication intercept.

The app uses Gemini to detect:

- **Leet speak** — like `w33d`, `c0caine`
- **Emoji steganography** — emojis used as drug code (❄️, 🍁, 💊, 🍄)
- **Multilingual slang** — Hinglish, Spanglish, mixed-language messages
- **Street slang** — from the local `slang_db.json` database

The result is one of four labels:
- `Illegal Drug Intent`
- `Drug Promotion`
- `Depression/Health`
- `Safe`

**Adaptive Learning**: If a new slang term is detected, you're given the option to add it to the database so the system learns it going forward.

In **Offline Mode**, a lightweight keyword heuristic replaces Gemini for on-field use without internet.

---

### 5. Audio Intelligence

Record audio directly in the browser or upload a `.wav`, `.mp3`, or `.m4a` file.

The app does:

- **Voice Biometrics (Simulation)** — generates a speaker fingerprint ID and matches it against a suspect database
- **Gemini Audio Analysis** — transcribes the audio and checks for drug-related conversations, identifies multiple speakers, and returns a threat score
- **RF Spectrum Sweeper** — runs an FFT (Fast Fourier Transform) on `.wav` files to detect high-frequency anomalies above 15kHz (useful for detecting hidden electronic bugs or ultrasonic beacons)

If a threat is detected, the evidence is logged and a Telegram alert is sent with the transcript.

---

### 6. Video Forensics

Upload an `.mp4`, `.avi`, or `.mov` file.

The app extracts frames at a configurable rate (0.5 to 2.0 FPS) and runs YOLO detection on each frame. All findings are logged in a table with:

- Timestamp within the video
- Detected object name
- Confidence score
- Frame number

You can then click "Log to Blockchain" to permanently store the video scan summary in the evidence chain.

---

### 7. 3D Scene Reconstruction

Upload a still image and the app generates a **3D point cloud** using estimated depth information. The result is a fully navigable 3D visualization rendered with Plotly — you can rotate, zoom, and pan the scene. This is inspired by the "Minority Report" style crime scene analysis concept.

---

### 8. Criminal Network Graph

GuardianAI reads your entire evidence chain and builds a visual graph showing relationships between:

- **People** (speaker IDs from audio analysis) — shown in red
- **Locations** (cities from IP geolocation) — shown in blue
- **Drugs / Weapons** (YOLO detection labels) — shown in purple / red

Edges connect entities that appeared in the same detection event, labeled with relationships like `sighted at`, `spotted in`, or `possession`. The graph uses Graphviz with a dark theme inspired by professional intel tools like Maltego.

---

### 9. Whistleblower Portal

Allows anyone to submit evidence **anonymously**:

1. Upload a file — it is hashed client-side
2. A simulated ZK-SNARK proof is generated (proving the file is valid without revealing its source)
3. The proof is "submitted to a DAO" and a Soulbound Token (SBT) is minted as confirmation

This module demonstrates how privacy-preserving evidence submission could work using Zero-Knowledge cryptography and blockchain-based bounty systems.

---

### 10. Evidence Chain (Blockchain Log)

Every detection from every module is stored in a local JSON-based evidence chain (`evidence_chain.json`). Each block contains:

- A **SHA256 hash** of its own content
- A reference to the **previous block's hash** (like a blockchain)
- The detection data, type, confidence, and GPS location
- A timestamp

If any block is modified manually, the chain verification will detect the mismatch and flag tampering. Each session gets its own file. Locked sessions are renamed with a `.LOCKED` extension and cannot be overwritten.

High-confidence threats also get:

- **ZK-Proof badges** — a short cryptographic proof string shown on the evidence card
- **Digital Warrant flow** — evidence above 80% confidence requires a simulated two-judge multi-signature approval before details are unlocked
- **Soulbound Tokens (SBTs)** — non-transferable blockchain tokens that record custody events (who viewed, when, what action was taken)
- **Publish to Sepolia** — one-click button to push evidence to the Ethereum Sepolia testnet

---

### 11. AI Copilot

The sidebar has a chatbox powered by Gemini. You can ask questions like:

- *"Summarize the last 5 threats"*
- *"Were any drugs detected in Mumbai?"*
- *"What was the highest confidence detection?"*

The Copilot receives the last 20 evidence blocks as context and responds in plain language.

---

### 12. Telegram Alerts

Whenever a threat is detected (text, image, webcam, audio), the app automatically sends a message to a Telegram bot that you configure. For image threats, the flagged image is attached. The message includes:

- Time of detection
- Location (city from IP)
- Reason / description of the threat

---

### 13. Ghost / Panic Mode

Press the **PANIC MODE** button in the sidebar. The entire app instantly transforms into a fake "Office Supply Inventory" management system (`IMS v2.4`) showing mundane items like paper reams and staplers. This is a decoy for situations where the operator needs to hide the real purpose of the app.

To exit Ghost Mode, type `phoenix` in the inventory search bar.

---

### 14. Dead Man's Switch

A background script (`security_daemon.py`) runs alongside the app. It writes a heartbeat timestamp every few seconds to a file called `.heartbeat`.

The main app reads this file on startup. If the heartbeat is more than 20 seconds old (meaning the daemon was killed), the app shows a full-screen red overlay: **"SYSTEM COMPROMISED"** and stops all operations. This prevents attackers from disabling the security layer without triggering a visible alarm.

---

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Web UI framework |
| **YOLOv8 (Ultralytics)** | Real-time object detection |
| **Google Gemini 2.5 Flash** | Multimodal AI (text, image, audio) |
| **OpenCV** | Image and video processing |
| **MediaPipe** | Pose estimation for behavioral analysis |
| **Plotly / PyDeck** | Interactive charts and 3D maps |
| **Graphviz** | Criminal network graph rendering |
| **python-dotenv** | Environment variable management |
| **feedparser** | RSS news feed parsing |
| **Telegram Bot API** | Real-time alert notifications |

---

## Project Structure

```
drug_detection_app/
├── app.py                      # Main application — all UI pages
├── behavior_engine.py          # Behavioral profiling (Psycho-Pass module)
├── scene_reconstruction.py     # 3D point cloud generation
├── tamper.py                   # Evidence chain, hashing, ZK-proofs, SBTs
├── swarm.py                    # Swarm networking (peer discovery)
├── security_daemon.py          # Dead Man's Switch background daemon
├── stega.py                    # Steganography utilities
├── slang_db.json               # Adaptive drug slang dictionary
├── style.css                   # Custom dark theme CSS
├── admin_secrets.yaml          # Admin configuration (keep private)
├── pose_landmarker_lite.task   # MediaPipe pose model file
├── yolo26n_finetuned.pt        # Custom-trained YOLO model (primary)
├── yolo26n.pt                  # Base YOLO model
├── yolov8n.pt                  # YOLOv8 nano model
├── yolo_third_phase.pt         # Third-phase fine-tuned YOLO model
├── requirements.txt            # Python dependencies
└── vault/                      # Secure file storage directory
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or newer
- A virtual environment (strongly recommended)
- A Google Gemini API key (free tier available at [aistudio.google.com](https://aistudio.google.com))
- (Optional) A Telegram Bot token and chat ID for alerts

### Step 1 — Clone or Download the Project

If you have the folder already, open a terminal and navigate into it:

```powershell
cd C:\Users\YourName\Downloads\drug_detection_app
```

### Step 2 — Create a Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3 — Install Dependencies

```powershell
pip install -r requirements.txt
```

> If you run into issues with `mediapipe` or `ultralytics`, make sure you have Visual C++ redistributables installed on Windows.

---

## Environment Variables

Create a file named `.env` in the root of the project folder. Add the following:

```env
# Required — Gemini API key from Google AI Studio
GEMINI_API_KEY_2=your_gemini_api_key_here

# Optional — Telegram bot for real-time alerts
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

**How to get a Gemini API key:**
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Paste it into the `.env` file above

**How to get a Telegram Bot token:**
1. Open Telegram and search for `@BotFather`
2. Type `/newbot` and follow the instructions
3. Copy the token provided
4. To get your Chat ID, message `@userinfobot` in Telegram

---

## Running the App

### Step 1 — Start the Security Daemon (optional but recommended)

Open a separate terminal window:

```powershell
python security_daemon.py
```

Keep this running in the background. It powers the Dead Man's Switch.

### Step 2 — Launch the Main App

```powershell
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Security Features

| Feature | Description |
|--------|-------------|
| **Tamper-Proof Evidence Chain** | SHA256-linked blocks — any modification is detected automatically |
| **Dead Man's Switch** | App halts if the security daemon is killed |
| **Ghost / Panic Mode** | One-click app disguise as fake inventory system |
| **Auto-Lock on Abandonment** | Console locks if the operator moves away for 30 seconds |
| **Privacy Blur** | Camera feed blurs if the operator's eyes leave the screen |
| **Digital Warrant System** | High-confidence evidence is sealed until multi-sig judicial approval |
| **ZK-Proofs** | Zero-Knowledge Proof badges on every high-risk evidence block |
| **Soulbound Tokens** | Immutable custody trail for every action taken on evidence |
| **Admin Voice Auth** | (Experimental) Voice-based admin authentication |
| **Offline Mode** | Full local operation without any external API calls |

---



## License

This project is for educational and research purposes. Always comply with local laws and regulations when deploying any surveillance or detection system.
