# utils/load_encoder.py

import os
import joblib

def load_encoder(load_path: str):
    """Load a fitted encoder from disk."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Encoder file not found at: {load_path}")
    
    encoder = joblib.load(load_path)
    print(f"✅ Encoder loaded from {load_path}")
    return encoder
