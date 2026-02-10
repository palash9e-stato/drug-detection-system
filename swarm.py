import time
import json
import os
import glob
import threading
import uuid
import shutil

# CONFIG
SWARM_DIR = ".swarm_beacon"
NODE_ID = f"{os.getenv('COMPUTERNAME', 'Guardian')}_{uuid.uuid4().hex[:4]}"

# Ensure swarm directory exists
if not os.path.exists(SWARM_DIR):
    try:
        os.makedirs(SWARM_DIR)
    except:
        pass

class SwarmNode:
    def __init__(self):
        self.running = True
        self.peers = {}
        self.cleanup_count = 0

    def start(self):
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        threading.Thread(target=self.scan_loop, daemon=True).start()

    def heartbeat_loop(self):
        """Writes a heartbeat file every 2 seconds"""
        my_beacon_file = os.path.join(SWARM_DIR, f"{NODE_ID}.json")
        
        while self.running:
            try:
                data = {
                    "id": NODE_ID,
                    "last_seen": time.time(),
                    "status": "active"
                }
                # Atomic write (write to temp then rename)
                tmp_file = my_beacon_file + ".tmp"
                with open(tmp_file, "w") as f:
                    json.dump(data, f)
                
                # Rename to atomic update
                if os.path.exists(my_beacon_file):
                    os.remove(my_beacon_file)
                os.rename(tmp_file, my_beacon_file)
                
                time.sleep(2)
            except Exception as e:
                print(f"Swarm Heartbeat Error: {e}")
                time.sleep(2)

    def scan_loop(self):
        """Scans the beacon directory for other nodes"""
        while self.running:
            try:
                active_peers = {}
                now = time.time()
                
                beacon_files = glob.glob(os.path.join(SWARM_DIR, "*.json"))
                
                for fpath in beacon_files:
                    try:
                        # Skip self? No, let's include all then filter in get_peers
                        fname = os.path.basename(fpath)
                        if fname == f"{NODE_ID}.json":
                            continue
                            
                        with open(fpath, "r") as f:
                            data = json.load(f)
                            
                        # Check age
                        if now - data["last_seen"] < 10:
                            active_peers[data["id"]] = data
                        else:
                            # Cleanup old files occasionally
                            if now - data["last_seen"] > 30:
                                try: os.remove(fpath)
                                except: pass
                                
                    except Exception:
                        continue
                        
                self.peers = active_peers
                time.sleep(1)
                
            except Exception as e:
                print(f"Swarm Scan Error: {e}")
                time.sleep(1)

    def get_active_peers(self):
        return list(self.peers.values())

# Singleton
swarm = SwarmNode()
