import subprocess
import time
import os
from pyngrok import ngrok, conf

# Set ngrok auth token
conf.get_default().auth_token = "35RBLsgHUffPiyRpdFVyRNboZ5J_4UJEsM6Ua2PUXDyVfzZGX"

# Kill any existing ngrok processes
try:
    ngrok.kill()
except:
    pass

# Create secure tunnel to port 8501
try:
    public_url = ngrok.connect(8501).public_url
    print(f"\n✅ Public URL: {public_url}\n")
    print(f"✅ Local URL: http://localhost:8501\n")
except Exception as e:
    print(f"Error setting up ngrok: {e}")

# Keep ngrok running
time.sleep(999999)
