# monitoring/anomaly_detector.py
import os
import joblib
import numpy as np
from numpy.linalg import pinv

ANOMALY_PATH = "artifacts/anomaly/mahalanobis.pkl"

def fit_mahalanobis(X, contamination=0.01):
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    cov_inv = pinv(cov)
    d2 = np.sum((X - mean) @ cov_inv * (X - mean), axis=1)
    thresh = np.percentile(d2, 100 * (1 - contamination))
    model = {"mean": mean.tolist(), "cov_inv": cov_inv.tolist(), "threshold": float(thresh)}
    os.makedirs(os.path.dirname(ANOMALY_PATH), exist_ok=True)
    joblib.dump(model, ANOMALY_PATH)
    return model

def load_mahalanobis(path=None):
    p = path or ANOMALY_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"Mahalanobis model not found: {p}")
    return joblib.load(p)

def mahalanobis_distance(sample, model):
    mean = np.asarray(model["mean"], dtype=float)
    cov_inv = np.asarray(model["cov_inv"], dtype=float)
    x = np.asarray(sample, dtype=float)
    d2 = float(np.sum((x - mean) @ cov_inv * (x - mean)))
    return d2

def is_anomalous(sample, model):
    d2 = mahalanobis_distance(sample, model)
    return d2 > model["threshold"]