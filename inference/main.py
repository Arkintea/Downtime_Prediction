# inference/main.py

import os
import joblib
import warnings
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
import uvicorn
from data_preprocessing.preprocessor import Preprocessor
from utils.s3_logger import push_log_to_s3
from monitoring.drift_detector import (
    initialise_drift_detector, 
    detect_feature_drift, 
    save_drift_detector, 
    ensure_drift_detector_initialized,
    update_reference_statistics
)
from monitoring.anomaly_detector import load_mahalanobis, is_anomalous
from monitoring.prometheus_metrics import (
    init_prometheus_server, 
    record_prediction, 
    record_drift_detection, 
    record_anomaly_detection, 
    update_model_health
)

warnings.filterwarnings("ignore")

MODEL_PATH = "artifacts/models/model.pkl"
SCALER_PATH = "artifacts/models/scaler.pkl"
LABEL_ENCODER_PATH = "encoders/label_encoder.pkl"
ONEHOT_COLUMNS_PATH = "encoders/onehot_columns.pkl"
FEATURE_NAMES_PATH = "encoders/feature_names.pkl"
SAVE_BATCH_SIZE = 100

os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/drift", exist_ok=True)
os.makedirs("artifacts/anomaly", exist_ok=True)
os.makedirs("encoders", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def load_model_components():
    try:
        required_files = [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, ONEHOT_COLUMNS_PATH, FEATURE_NAMES_PATH]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
            
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoders = joblib.load(LABEL_ENCODER_PATH)
        downtime_label_encoder = label_encoders["Downtime_Group"]
        onehot_columns = joblib.load(ONEHOT_COLUMNS_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        
        print("Model components loaded successfully")
        return model, scaler, downtime_label_encoder, onehot_columns, feature_names
        
    except FileNotFoundError as e:
        print(f"Missing required files: {e}")
        print("Run training pipeline first: python training_pipeline/train_pipeline.py")
        raise e
    except Exception as e:
        print(f"Error loading components: {e}")
        raise e

model, scaler, downtime_label_encoder, onehot_columns, feature_names = load_model_components()

drift_detectors, reference_stats = None, None
anomaly_detector = None
drift_enabled = False
prediction_counter = 0

try:
    drift_detectors, reference_stats = initialise_drift_detector()
    if drift_detectors is not None and reference_stats is not None:
        drift_enabled = True
    else:
        print("Drift detectors not found - will initialise on first prediction")
except Exception as e:
    print(f"Could not load drift detectors: {e}")

try:
    anomaly_detector = load_mahalanobis()
    print("Anomaly detector loaded successfully")
except Exception as e:
    print(f"Could not load anomaly detector: {e}")

app = FastAPI(title="Machine Downtime Prediction API", version="1.0.0")

try:
    init_prometheus_server(port=8001)
    update_model_health(True)
except Exception as e:
    print(f"Prometheus initialization failed: {e}")

batch_log = []
prediction_results = []

class InferenceRequest(BaseModel):
    machine_id: str
    assembly_line_no: str
    data: Dict[str, Any]

@app.get("/")
async def root():
    return {
        "message": "Machine Downtime Prediction API", 
        "version": "1.0.0",
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "encoder_loaded": downtime_label_encoder is not None,
        "model_loaded": model is not None,
        "drift_detection_enabled": drift_enabled,
        "anomaly_detection_enabled": anomaly_detector is not None
    }

@app.post("/predict")
async def predict(request: InferenceRequest):
    global drift_detectors, reference_stats, drift_enabled, prediction_counter
    
    try:
        if drift_detectors is None or reference_stats is None:
            drift_detectors, reference_stats = ensure_drift_detector_initialized(feature_names)
            drift_enabled = True

        df = pd.DataFrame([request.data])
        df["Machine_ID"] = request.machine_id
        df["Assembly_Line_No"] = request.assembly_line_no

        df = Preprocessor.reconstruct_datetime(df)
        df = Preprocessor.add_datetime_features(df)
        df = Preprocessor.add_temporal_features(df)

        drop_cols = ["Downtime", "Downtime_Group", "Future_Downtime_Label"]
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors="ignore")

        df_processed = Preprocessor.one_hot_encode_features(df, columns=["Machine_ID", "Assembly_Line_No"], save_path=None)
        df_processed = Preprocessor.align_one_hot_columns(df_processed, onehot_columns)
        df_processed = df_processed.reindex(columns=feature_names, fill_value=0)
        
        for col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        df_processed = df_processed.fillna(0).replace([np.inf, -np.inf], 0)
        
        X_scaled = scaler.transform(df_processed)
        feature_values = X_scaled[0]

        drift_results = detect_feature_drift(drift_detectors, feature_values, feature_names)
        
        is_anomaly = False
        if anomaly_detector:
            try:
                is_anomaly = is_anomalous(feature_values, anomaly_detector)
            except Exception:
                pass

        prediction_counter += 1
        
        if prediction_counter % 50 == 0:
            try:
                update_reference_statistics(reference_stats, feature_values, feature_names)
                save_drift_detector(drift_detectors, reference_stats)
            except Exception:
                pass

        pred_index = model.predict(X_scaled)[0]
        pred_proba = model.predict_proba(X_scaled)[0]
        
        try:
            pred_label = downtime_label_encoder.inverse_transform([pred_index])[0]
        except (ValueError, IndexError):
            pred_label = f"Unknown_Class_{pred_index}"
        
        confidence = float(max(pred_proba))

        record_prediction(confidence)
        record_drift_detection(drift_results['drifted_features'])
        record_anomaly_detection(is_anomaly)

        probabilities = {}
        for i, prob in enumerate(pred_proba):
            try:
                class_name = downtime_label_encoder.inverse_transform([i])[0]
                probabilities[str(class_name)] = float(prob)
            except (ValueError, IndexError):
                probabilities[f"Class_{i}"] = float(prob)

        result = {
            "machine_id": str(request.machine_id),
            "assembly_line_no": str(request.assembly_line_no),
            "predicted_downtime": str(pred_label),
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat(),
            "probabilities": probabilities
        }

        prediction_results.append(result)
        if len(prediction_results) > 1000:
            prediction_results.pop(0)

        log_entry = dict(request.data)
        current_time = datetime.now()
        log_entry["Date"] = current_time.strftime("%Y-%m-%d")
        log_entry["Timestamp"] = current_time.strftime("%H:%M:%S")
        log_entry["Machine_ID"] = request.machine_id
        log_entry["Assembly_Line_No"] = request.assembly_line_no
        log_entry["Downtime"] = pred_label
        log_entry["Is_Anomaly"] = is_anomaly
        log_entry["Drift_Detected"] = drift_results['drift_detected']

        batch_log.append(log_entry)

        if len(batch_log) >= SAVE_BATCH_SIZE:
            try:
                df_log = pd.DataFrame(batch_log)
                push_log_to_s3(df_log)
                batch_log.clear()
            except Exception:
                pass

        return result

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        raise HTTPException(status_code=400, detail=error_msg)

@app.get("/latest_predictions")
async def get_latest_predictions():
    return {"predictions": prediction_results[-100:]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)