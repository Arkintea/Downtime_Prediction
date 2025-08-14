# utils/save_encoder.py

import os
import joblib

def save_encoder(encoders: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(encoders, save_path)
