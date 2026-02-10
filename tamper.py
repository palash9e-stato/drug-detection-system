"""
Tamper-Proof Evidence Storage Module
Implements basic tamper detection using hashing and blockchain-like chains
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path

EVIDENCE_FILE = "evidence_chain.json"

def compute_hash(data):
    """Compute SHA256 hash of data"""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()

def load_chain():
    """Load evidence chain from file"""
    if Path(EVIDENCE_FILE).exists():
        try:
            with open(EVIDENCE_FILE, 'r') as f:
                content = f.read().strip()
                if content:  # Only parse if file has content
                    return json.loads(content)
        except (json.JSONDecodeError, IOError):
            pass  # File is corrupted, start fresh
    return {"chain": [], "last_hash": "0"}

def save_chain(chain):
    """Save evidence chain to file"""
    with open(EVIDENCE_FILE, 'w') as f:
        json.dump(chain, f, indent=2)

def verify_chain():
    """Verify integrity of entire evidence chain"""
    chain = load_chain()
    last_hash = "0"
    
    for i, block in enumerate(chain["chain"]):
        expected_prev_hash = last_hash
        actual_prev_hash = block.get("previous_hash")
        
        if actual_prev_hash != expected_prev_hash:
            return False, f"Tampering detected at block {i}: hash mismatch"
        
        # Recompute current block hash
        block_data = {
            "timestamp": block["timestamp"],
            "type": block["type"],
            "data": block["data"],
            "previous_hash": block["previous_hash"]
        }
        if "severity" in block:
            block_data["severity"] = block["severity"]
        expected_hash = compute_hash(block_data)
        actual_hash = block.get("hash")
        
        if actual_hash != expected_hash:
            return False, f"Tampering detected at block {i}: content modified"
        
        last_hash = actual_hash
    
    return True, "Chain integrity verified ✓"

def store_evidence(data):
    """
    Store evidence in tamper-proof chain with advanced forensic metadata
    
    Args:
        data (dict): Evidence data to store. Should include 'meta' dict if possible.
        
    Returns:
        dict: Block info with hash and timestamp
    """
    chain = load_chain()
    
    # Auto-calculate Severity
    conf = data.get("confidence", 0)
    label = data.get("label", "").lower()
    severity = "LOW"
    
    if "drug" in label or "weapon" in label or "threat" in label:
        if conf > 0.8: severity = "CRITICAL"
        elif conf > 0.5: severity = "HIGH"
        else: severity = "MEDIUM"
    
    # Ensure Meta exists
    if "meta" not in data:
        data["meta"] = {}
        
    # Inject Default Meta if missing
    data["meta"].setdefault("app_version", "3.0-CYBER")
    data["meta"].setdefault("captured_at", datetime.utcnow().isoformat())
    data["severity"] = severity
    
    # Create new block
    block = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": data.get("type", "unknown"),
        "severity": severity,
        "data": data,
        "previous_hash": chain["last_hash"]
    }
    
    # Compute block hash (including new fields)
    block_hash = compute_hash(block)
    block["hash"] = block_hash
    
    # Add to chain
    chain["chain"].append(block)
    chain["last_hash"] = block_hash
    
    # Save
    save_chain(chain)
    
    return {
        "success": True,
        "hash": block_hash,
        "timestamp": block["timestamp"],
        "severity": severity,
        "block_index": len(chain["chain"]) - 1
    }

def get_evidence(block_index=None):
    """
    Retrieve evidence from chain
    
    Args:
        block_index (int): Specific block index, or None to get all
        
    Returns:
        dict or list: Evidence block(s)
    """
    chain = load_chain()
    
    if block_index is None:
        return chain["chain"]
    
    if 0 <= block_index < len(chain["chain"]):
        return chain["chain"][block_index]
    
    return None

def get_evidence_count():
    """Get total number of evidence blocks stored"""
    chain = load_chain()
    return len(chain["chain"])

def export_chain(filename="exported_chain.json"):
    """Export entire chain for verification"""
    chain = load_chain()
    with open(filename, 'w') as f:
        json.dump(chain, f, indent=2)
    return filename

def publish_to_chain(block_index=-1):
    """
    Simulates publishing an evidence block to the Ethereum Sepolia Testnet.
    Returns a mock transaction hash.
    """
    import time
    import random
    
    chain = load_chain()
    blocks = chain.get("chain", [])
    
    if not blocks:
        return None
        
    # Get target block
    target = blocks[block_index]
    
    # Check if already on-chain
    if "blockchain_tx" in target:
        return target["blockchain_tx"]
        
    # Simulate Network Delay
    time.sleep(1.5)
    
    # Generate Mock TX Hash
    tx_hash = "0x" + hashlib.sha256(str(random.random()).encode()).hexdigest()
    
    # Update Record
    target["blockchain_tx"] = tx_hash
    target["blockchain_network"] = "Sepolia (Testnet)"
    target["blockchain_time"] = datetime.utcnow().isoformat()
    
    save_chain(chain)
    return tx_hash

# -----------------------------
# FUTURISTIC BLOCKCHAIN FEATURES
# -----------------------------

class SmartContract:
    """Mock Smart Contract for Digital Warrants"""
    def __init__(self):
        self.state = "LOCKED"
        self.approvals = []
        self.required_approvals = 2
        
    def request_warrant(self, evidence_id):
        """Broadcast a warrant request event"""
        return {
            "event": "WARRANT_REQUESTED",
            "evidence_id": evidence_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "PENDING_JUDICIAL_REVIEW"
        }
    
    def approve(self, judge_key):
        """Simulate a judicial approval (multi-sig)"""
        if judge_key not in self.approvals:
            self.approvals.append(judge_key)
            
        if len(self.approvals) >= self.required_approvals:
            self.state = "UNLOCKED"
            return True
        return False

def generate_zk_proof(data):
    """
    Generates a Zero-Knowledge Proof (simulation).
    Proves: Confidence > 0.8 AND Class in [Drugs, Weapons]
    Without revealing: The actual class or image.
    """
    import random
    
    # 1. Private Inputs
    conf = data.get("confidence", 0)
    label = data.get("label", "")
    
    # 2. Circuit Logic (Mock)
    is_valid = conf > 0.8 and ("drug" in label or "weapon" in label)
    
    if is_valid:
        # Generate a convincing ZK-SNARK string
        proof_hash = hashlib.sha256(f"{conf}{label}{random.random()}".encode()).hexdigest()[:32]
        return f"zk-snark-proof:0x{proof_hash}..."
    return None

def mint_sbt(user_id, action, evidence_id):
    """
    Mints a Soulbound Token (SBT) to the user's wallet.
    Logs immutable chain of custody.
    """
    sbt = {
        "token_id": f"SBT-{hashlib.sha256(str(datetime.utcnow()).encode()).hexdigest()[:8]}",
        "holder": user_id,
        "action": action, # VIEW, EXPORT, SHARE
        "evidence_ref": evidence_id,
        "timestamp": datetime.utcnow().isoformat(),
        "attributes": ["Non-Transferable", "Audit-Log"]
    }
    
    # In a real app, this goes to a separate ledger.
    # Here, we append to a 'custody_log.json' or just return it for the UI.
    return sbt
