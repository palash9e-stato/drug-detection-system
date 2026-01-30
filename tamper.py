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
        expected_hash = compute_hash(block_data)
        actual_hash = block.get("hash")
        
        if actual_hash != expected_hash:
            return False, f"Tampering detected at block {i}: content modified"
        
        last_hash = actual_hash
    
    return True, "Chain integrity verified ✓"

def store_evidence(data):
    """
    Store evidence in tamper-proof chain
    
    Args:
        data (dict): Evidence data to store
        
    Returns:
        dict: Block info with hash and timestamp
    """
    chain = load_chain()
    
    # Create new block
    block = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": data.get("type", "unknown"),
        "data": data,
        "previous_hash": chain["last_hash"]
    }
    
    # Compute block hash
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
