import subprocess
import time
from pyngrok import ngrok, conf

# Set ngrok auth token
conf.get_default().auth_token = "35RBLsgHUffPiyRpdFVyRNboZ5J_4UJEsM6Ua2PUXDyVfzZGX"

# Kill any existing ngrok processes
try:
    ngrok.kill()
except:
    pass

# Create secure tunnel to port 8501
public_url = ngrok.connect(8501).public_url
print(f"\n🌐 Public URL: {public_url}\n")

# Run streamlit with specific parameters
subprocess.run([
    "streamlit", "run", "app.py",
    "--server.port", "8501",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false"
])
