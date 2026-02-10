import time
import json
import shutil
import os
import hashlib
import requests
import cv2
import random
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from tamper import verify_chain, load_chain
from stega import encode_lsb

# Load Environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_2")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Config
EVIDENCE_FILE = "evidence_chain.json"
SLANG_DB_FILE = "slang_db.json"
CANARY_FILE = "admin_secrets.yaml"
HEARTBEAT_FILE = ".heartbeat"
BACKUP_FOLDER = "vault"  # Hidden vault for stega images

CHECK_INTERVAL = 1  # Check file every 1 second
SLANG_UPDATE_INTERVAL = 300  # Update slang every 5 minutes
STEGA_BACKUP_INTERVAL = 300 # Backup evidence to stats every 5 mins

# Create Vault
if not os.path.exists(BACKUP_FOLDER):
    os.makedirs(BACKUP_FOLDER)

def send_telegram_alert(message, image_path=None):
    """Send alert to Telegram Bot"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[WARN] Telegram Token missing. Alert would be: {message}")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🚨 GUARDIAN SECURITY ALERT 🚨\n\n{message}"
    }
    
    try:
        requests.post(url, json=payload, timeout=5)
        
        # Send Photo if available
        if image_path and os.path.exists(image_path):
            photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, 'rb') as f:
                files = {'photo': f}
                requests.post(photo_url, data={"chat_id": TELEGRAM_CHAT_ID}, files=files, timeout=10)
            
        return True
    except Exception as e:
        print(f"[ERR] Telegram Error: {e}")
        return False

def capture_intruder():
    """Captures a webcam frame of the intruder."""
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            filename = f"intruder_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            return filename
    except:
        return None
    return None

class SecurityDaemon:
    def __init__(self):
        self.last_known_good_chain = None
        self.last_slang_update = 0
        self.last_stega_backup = 0
        
        # Canary State
        self.canary_mtime = 0
        if os.path.exists(CANARY_FILE):
             self.canary_mtime = os.stat(CANARY_FILE).st_mtime
        
        # Initialize Gemini Client
        self.client = None
        if GEMINI_API_KEY:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
                print("[OK] Gemini Client Connected for Security Operations")
            except Exception as e:
                print(f"[ERR] Gemini Connection Failed: {e}")
        else:
            print("[WARN] GEMINI_API_KEY_2 not found. Slang updates disabled.")

        # Initialize State
        self.load_initial_state()

    def load_initial_state(self):
        """Load the initial valid state of the evidence chain."""
        print("[INIT] Initializing Golden State...")
        if os.path.exists(EVIDENCE_FILE):
            valid, msg = verify_chain()
            if valid:
                with open(EVIDENCE_FILE, 'r') as f:
                    self.last_known_good_chain = f.read()
                print("[OK] Evidence Chain Verified & Secured in Memory.")
            else:
                print(f"[WARN] Start-up Integrity Check Failed: {msg}")
                with open(EVIDENCE_FILE, 'r') as f:
                    self.last_known_good_chain = f.read()
        else:
            print("[WARN] No evidence file found. Waiting for creation.")
            self.last_known_good_chain = None

    def check_integrity(self):
        """Check if file has changed and if it's valid."""
        if not os.path.exists(EVIDENCE_FILE):
            return

        try:
            with open(EVIDENCE_FILE, 'r') as f:
                current_content = f.read()
            
            # If content matches our memory, no changes.
            if self.last_known_good_chain and current_content == self.last_known_good_chain:
                return

            print("[INFO] Change Detected in Evidence Chain...")
            
            valid, msg = verify_chain()
            
            if valid:
                print("[OK] Change Authorized. Updating Golden State.")
                self.last_known_good_chain = current_content
            else:
                print(f"[ALERT] TAMPERING DETECTED! {msg}")
                print("[ACTION] ACTIVATING AUTO-RESTORE PROTOCOL...")
                self.restore_backup()
                
                # Alert
                intruder_img = capture_intruder()
                msg = f"TAMPERING DETECTED on Evidence Chain!\nReason: {msg}\nAction: File Auto-Restored."
                send_telegram_alert(msg, intruder_img)

        except Exception as e:
            print(f"[ERR] Monitor Error: {e}")

    def restore_backup(self):
        """Overwrite the file with the last known good state."""
        if self.last_known_good_chain:
            try:
                with open(EVIDENCE_FILE, 'w') as f:
                    f.write(self.last_known_good_chain)
                print("[OK] SYSTEM RESTORED: Evidence file reverted to safe state.")
            except Exception as e:
                print(f"[ERR] Restoration Failed: {e}")

    def check_canary(self):
        """Monitors the Digital Canary (Honeypot) file."""
        if not os.path.exists(CANARY_FILE):
            return 
            
        try:
            current_mtime = os.stat(CANARY_FILE).st_mtime
            if current_mtime != self.canary_mtime:
                print("[ALERT] 🐥 DIGITAL CANARY TRIGGERED!")
                self.canary_mtime = current_mtime 
                
                # Capture & Alert
                intruder_img = capture_intruder()
                send_telegram_alert("DIGITAL CANARY TRIGGERED!\nSomeone accessed 'admin_secrets.yaml'.\nPotentially hostile actor.", intruder_img)
        except Exception as e:
            print(f"[ERR] Canary Check Error: {e}")

    def heartbeat(self):
        """Update dead man's switch heartbeat."""
        try:
            with open(HEARTBEAT_FILE, 'w') as f:
                f.write(str(datetime.utcnow().timestamp()))
        except:
            pass

    def perform_stega_backup(self):
        """Backs up evidence chain into an image."""
        if time.time() - self.last_stega_backup < STEGA_BACKUP_INTERVAL:
            return
            
        print("[INFO] Performing Steganographic Backup...")
        if not self.last_known_good_chain:
            return

        try:
            # 1. Capture a dummy frame to use as cover
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                # Create a black image if no cam
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            cover_path = os.path.join(BACKUP_FOLDER, "cover.png")
            cv2.imwrite(cover_path, frame)
            
            # 2. Encode
            backup_path = os.path.join(BACKUP_FOLDER, f"evidence_backup_{int(time.time())}.png")
            success = encode_lsb(cover_path, self.last_known_good_chain, backup_path)
            
            if success:
                print(f"[OK] Evidence successfully hidden in {backup_path}")
                # Cleanup cover
                if os.path.exists(cover_path): os.remove(cover_path)
            else:
                print("[WARN] Steganography failed (Data likely too large for image).")
                
        except Exception as e:
            print(f"[ERR] Stega Backup Error: {e}")
            
        self.last_stega_backup = time.time()

    def update_slang_dictionary(self):
        """Fetch new slang terms from Gemini."""
        if not self.client or (time.time() - self.last_slang_update < SLANG_UPDATE_INTERVAL):
            return

        print("[INFO] Scouring for new slang terms...")
        try:
            # Load current DB
            current_terms = []
            if os.path.exists(SLANG_DB_FILE):
                with open(SLANG_DB_FILE, 'r') as f:
                    db = json.load(f)
                    current_terms = db.get("terms", [])
            
            # Ask Gemini
            prompt = f"""
            List 5 trending or common street slang terms for illegal drugs that might not be in a standard dictionary.
            Return a JSON list of strings only, e.g. ["term1", "term2"].
            Do NOT include these existing terms: {current_terms[:20]}...
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=prompt
            )
            
            # extraction
            import re
            match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if match:
                new_terms = json.loads(match.group(0))
                
                # Update DB
                updated = False
                for term in new_terms:
                    term = term.lower()
                    if term not in current_terms:
                        current_terms.append(term)
                        updated = True
                        print(f"[NEW] Learned new slang: {term}")
                
                if updated:
                    with open(SLANG_DB_FILE, 'w') as f:
                        json.dump({
                            "version": "1.2-AUTO", 
                            "terms": current_terms, 
                            "last_updated": datetime.now().isoformat()
                        }, f, indent=2)
                    print("[OK] Slang Database Updated.")
            
        except Exception as e:
            print(f"[WARN] Slang Update Error: {e}")
            
        self.last_slang_update = time.time()

    def run(self):
        print(f"[INFO] GUARDIAN SECURITY DAEMON ACTIVE")
        print(f"   - Watch Target: {os.path.abspath(EVIDENCE_FILE)}")
        print(f"   - Canary: {CANARY_FILE}")
        print(f"   - Dead Man Switch: ACTIVE")
        print("------------------------------------------------")
        
        while True:
            self.check_integrity()
            self.check_canary()
            self.check_dead_hand()
            self.update_slang_dictionary()
            self.perform_stega_backup()
            self.heartbeat()
            time.sleep(CHECK_INTERVAL)

    def check_dead_hand(self):
        """Checks for operator life signals (operator_pulse). Wipes if dead."""
        OPERATOR_PULSE = ".operator_pulse"
        if not os.path.exists(OPERATOR_PULSE):
            return

        try:
            # Check last modification time
            mtime = os.path.getmtime(OPERATOR_PULSE)
            if time.time() - mtime > 3600: # 1 Hour
                print("[CRITICAL] HEAD HAND SWITCH TRIGGERED: OPERATOR MISSING")
                print("[ACTION] INITIATING EVIDENCE PURGE PROTOCOL...")
                
                if os.path.exists(EVIDENCE_FILE):
                     # Rename to lock
                     locked_name = f"{EVIDENCE_FILE}.LOCKED"
                     shutil.move(EVIDENCE_FILE, locked_name)
                     print(f"[OK] EVIDENCE VAULT SEALED: {locked_name}")
                     
                     # Alert (if still possible)
                     send_telegram_alert("💀 DEAD HAND SWITCH TRIGGERED\n\nOperator heartbeat lost for > 60 mins.\nEvidence chain has been SEALED and LOCKDOWN initiated.")
                     
                     # Create new empty chain to prevent crash but show empty
                     with open(EVIDENCE_FILE, 'w') as f:
                         json.dump({"chain": [], "status": "WIPED"}, f)
                
                # Delete Pulse file to reset? 
                os.remove(OPERATOR_PULSE)
                exit(0) # Kill Daemon
        except Exception as e:
            print(f"[ERR] Dead Hand Error: {e}")

if __name__ == "__main__":
    daemon = SecurityDaemon()
    daemon.run()
