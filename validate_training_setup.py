# validate_training_setup.py

import os
import joblib
import pandas as pd

def validate_training_artifacts():
    """Validate that all required artifacts exist after training"""
    required_files = {
        "Model": "artifacts/models/model.pkl",
        "Scaler": "artifacts/models/scaler.pkl", 
        "Label Encoder": "encoders/label_encoder.pkl",
        "One-hot Columns": "encoders/onehot_columns.pkl",
        "Feature Names": "encoders/feature_names.pkl",
        "Drift Detector": "artifacts/drift/drift_detector.pkl",
        "Reference Stats": "artifacts/drift/reference_stats.pkl",
        "Anomaly Detector": "artifacts/anomaly/mahalanobis.pkl",
        "Training Data": "data/splits/training_data.csv",
        "Inference Data": "data/splits/inference_data.csv"
    }
    
    print("🔍 VALIDATING TRAINING ARTIFACTS")
    print("=" * 50)
    
    missing_files = []
    for name, path in required_files.items():
        if os.path.exists(path):
            try:
                if path.endswith('.pkl'):
                    artifact = joblib.load(path)
                    print(f"✅ {name}: {path}")
                    
                    if name == "Model":
                        print(f"   └── Type: {type(artifact).__name__}")
                    elif name == "Feature Names":
                        print(f"   └── Features: {len(artifact)}")
                    elif name == "Label Encoder":
                        encoder = artifact["Downtime_Group"]
                        print(f"   └── Classes: {list(encoder.classes_)}")
                    elif name == "Anomaly Detector":
                        print(f"   └── Threshold: {artifact.get('threshold', 'N/A')}")
                elif path.endswith('.csv'):
                    df = pd.read_csv(path)
                    print(f"✅ {name}: {path}")
                    print(f"   └── Shape: {df.shape}")    
            except Exception as e:
                print(f"❌ {name}: {path} (Error loading: {e})")
        else:
            print(f"❌ {name}: {path} (Missing)")
            missing_files.append(path)
    
    print("=" * 50)
    
    if missing_files:
        print(f"❌ VALIDATION FAILED - {len(missing_files)} files missing")
        print("Please run: python training_pipeline/train_pipeline.py")
        return False
    else:
        print("✅ VALIDATION SUCCESSFUL - All artifacts present")
        return True

if __name__ == "__main__":
    validate_training_artifacts()