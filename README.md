# 🚨 GuardianAI: Omniscient Threat Detection Platform
### *Built for India Innovates Hackathon 2026*

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/AI-YOLOv8%20%7C%20Gemini%202.5-orange.svg" alt="AI Models">
  <img src="https://img.shields.io/badge/Security-Blockchain%20Hashing-green.svg" alt="Security">
</div>

---

## 🎯 The Problem We Are Solving

Criminal networks in India are evolving. They no longer rely solely on physical handoffs. Modern trafficking involves:
1. **Digital Subterfuge:** Using emoji codes (❄️, 🍁) and slang to disguise illicit communications.
2. **Coordinated Networks:** Operations span across multiple chat apps, physical borders, and secure drop-offs.
3. **Evidence Tampering:** Digital evidence can be easily modified, leading to dismissed court cases.

Current law enforcement tools are completely **siloed**. CCTVs don't talk to text forensics tools, and neither can guarantee that their digital evidence hasn't been altered. 

**GuardianAI bridges this gap.**

---

## 🚀 What is GuardianAI?

**GuardianAI is an all-in-one, AI-driven command center that unifies physical surveillance and digital forensics.** 

Designed specifically for the high-stakes, fast-paced environment of modern security operations, it analyzes **video, audio, and text simultaneously** to detect narcotics, weapons, and coded threats. 

Most importantly, it logs *every* detection into an unbreakable, cryptographic blockchain, ensuring 100% court admissibility.

---

## 💎 Innovating for India (Key Features)

### 1. 🧠 Multimodal "Omniscient" Detection
We don't just look for guns and drugs on a camera. GuardianAI runs two parallel models:
* **YOLOv8** (Custom Trained locally) for lightning-fast object detection. 
  * 📎 [View our YOLOv8 Fine-Tuning Kaggle Notebook](https://www.kaggle.com/code/pstomar/yolo26n-finetuned/edit)
  * 📎 [Read the Original YOLOv8 Architecture Paper](https://arxiv.org/abs/2304.00501)
* **Google Gemini 2.5 Flash** for deep multimodal reasoning. It reads intercepted text messages for regional street slang, analyzes audio spectrums for anomalies, and interprets the contextual "intent" behind images.

### 2. 🛡️ Unbreakable Chain of Custody (Digital Warrants)
Every piece of evidence goes into our local JSON blockchain. Each block is **SHA-256 hashed** and cryptographically chained to the previous one. If a single pixel of an image is tampered with by a rogue agent, the entire chain throws a "Tampered Evidence" alert. 

### 3. 🚨 Operator Privacy & "Ghost Mode"
* **Privacy Blur (Eyes-On-Glass):** The live webcam feed violently blurs if the operator looks away, preventing unauthorized personnel from snooping on classified streams.
* **Telegram Alerts:** High-confidence threats trigger instant notifications with images sent straight to command via the [Telegram Bot API](https://core.telegram.org/bots/api).
* **Panic / Ghost Mode:** If compromised, one click transforms the entire interface into a boring "Office Supply Inventory System."
* **Dead Man's Switch:** A background daemon constantly pulses the system. If the app is silently killed by an attacker, the system throws a full-screen lockdown.

### 4. 🗺️ Predictive "Pre-Crime" Mapping
GuardianAI doesn't just react; it predicts. By analyzing the geospatial data of past detections, it plots a **3D interactive hot-zone map** and dynamically draws a Maltego-style criminal network graph connecting suspects, drugs, and locations. 

### 5. 📴 Air-Gapped Operation
For field agents operating deep in areas with zero internet connectivity, GuardianAI’s **Offline Mode** seamlessly swaps out cloud APIs for lightweight local heuristic models.

---

## 🛠️ How it Works (Under the Hood)

1. **Dashboard:** The central nervous system showing threat stats, system uptime, and the 3D map.
2. **Visual/Live Forensics:** Feed in images, videos, or IP camera streams. The YOLO+Gemini combo scans every frame. 
3. **Semantic/Audio Intelligence:** Paste text or upload audio recordings. It translates and flags leet-speak (`c0caine`) and local slang from the `slang_db.json`. 
4. **Whistleblower Portal:** Allows anonymous tipping using zero-knowledge (ZK) simulated proofs, a first for citizen-reporting in India.

---

## ⚙️ Try It Yourself (Installation)

### Prerequisites
* Python 3.10+
* [Google Gemini API Key](https://aistudio.google.com/)

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_REPO_NAME/GuardianAI.git
cd GuardianAI

# 2. Setup your virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your environment
# Rename .env.example to .env and add your Gemini API Key

# 5. Ignite the Security Daemon (Background dead-man switch)
python security_daemon.py

# 6. Launch the Command Center 
streamlit run app.py
```

---

## 🔮 The Future Roadmap
Post-hackathon, we aim to integrate GuardianAI directly with state-level drones for aerial YOLO detection and deploy federated learning so multiple police departments can share threat data without sharing raw, sensitive images.

> *"Because the future of national security isn't just about watching; it's about understanding."*

---

## 👥 The Team

* **Palash Singh Tomar** — [LinkedIn](https://www.linkedin.com/in/palash-singh-tomar-14442a386/)
* **Namrata Tiwari** — [LinkedIn](https://www.linkedin.com/in/namrata-tiwari-163942370/)
* **Mayuri Pandey** — [LinkedIn](https://www.linkedin.com/in/mayuri-pandey-75598835a/)
* **Parul Vyas** — [LinkedIn](https://www.linkedin.com/in/parul-vyas-313839362/)
* **Mohammed Arhaan** — [LinkedIn](https://www.linkedin.com/in/mohammed-arhaan-78986627b/)
* **Payal Verma** — [LinkedIn](https://www.linkedin.com/in/payal-verma-2692a5363/)
