# monitoring/drift_detector.py
import os
import joblib
import numpy as np
from river import drift
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

DRIFT_DETECTOR_PATH = "artifacts/drift/drift_detector.pkl"
REFERENCE_STATS_PATH = "artifacts/drift/reference_stats.pkl"

def initialise_drift_detector(feature_names: Optional[List[str]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if feature_names is None:
        if os.path.exists(DRIFT_DETECTOR_PATH) and os.path.exists(REFERENCE_STATS_PATH):
            try:
                detectors = joblib.load(DRIFT_DETECTOR_PATH)
                reference_stats = joblib.load(REFERENCE_STATS_PATH)
                print(f"Loaded existing drift detectors from {DRIFT_DETECTOR_PATH}")
                return detectors, reference_stats
            except Exception as e:
                print(f"Failed to load existing drift detectors: {e}")
                return None, None
        else:
            return None, None
    
    detectors = {}
    for feature in feature_names:
        detectors[feature] = drift.ADWIN(delta=0.01)
    
    print(f"Created new drift detectors for {len(feature_names)} features")
    return detectors, None

def save_drift_detector(detectors: Dict[str, Any], reference_stats: Dict[str, Any]):
    os.makedirs(os.path.dirname(DRIFT_DETECTOR_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(REFERENCE_STATS_PATH), exist_ok=True)
    
    joblib.dump(detectors, DRIFT_DETECTOR_PATH)
    joblib.dump(reference_stats, REFERENCE_STATS_PATH)
    print(f"Drift detector saved to {DRIFT_DETECTOR_PATH}")

def ensure_drift_detector_initialized(feature_names: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    detectors, reference_stats = initialise_drift_detector()
    
    if detectors is None or reference_stats is None:
        print("Drift detectors not found, initialising new ones...")
        
        detectors, _ = initialise_drift_detector(feature_names)
        
        reference_stats = {feature: {
            'mean': 0.0,
            'std': 1.0,
            'min': 0.0,
            'max': 1.0,
            'count': 0
        } for feature in feature_names}
        
        save_drift_detector(detectors, reference_stats)
        print("New drift detectors initialised and saved")
    
    missing_features = []
    for feature in feature_names:
        if feature not in detectors:
            detectors[feature] = drift.ADWIN(delta=0.01)
            missing_features.append(feature)
    
    if missing_features:
        print(f"Added drift detectors for missing features: {missing_features}")
        save_drift_detector(detectors, reference_stats)
    
    return detectors, reference_stats

def detect_feature_drift(detectors: Dict[str, Any], feature_values: np.ndarray, 
                        feature_names: List[str]) -> Dict[str, Any]:
    if len(feature_values) != len(feature_names):
        raise ValueError(f"Feature values length ({len(feature_values)}) does not match feature names length ({len(feature_names)})")
    
    drift_results = {
        'drift_detected': False,
        'drifted_features': [],
        'drift_scores': {}
    }
    
    for i, feature_name in enumerate(feature_names):
        detector = detectors.get(feature_name)
        if detector is None:
            continue
            
        try:
            feature_value = float(feature_values[i])
            
            if np.isnan(feature_value) or np.isinf(feature_value):
                feature_value = 0.0
            
            was_drifted = getattr(detector, 'drift_detected', False)
            
            detector.update(feature_value)
            
            current_drift = detector.drift_detected
            
            if current_drift and not was_drifted:
                drift_results['drift_detected'] = True
                drift_results['drifted_features'].append(feature_name)
            
            drift_results['drift_scores'][feature_name] = {
                'current_value': feature_value,
                'drift_detected': current_drift,
                'variance': getattr(detector, 'variance', 0.0) if hasattr(detector, 'variance') else 0.0
            }
            
        except Exception as e:
            print(f"Error processing feature {feature_name}: {e}")
            drift_results['drift_scores'][feature_name] = {
                'current_value': 0.0,
                'drift_detected': False,
                'error': str(e)
            }
    
    return drift_results

def update_reference_statistics(reference_stats: Dict[str, Any], feature_values: np.ndarray, 
                               feature_names: List[str], alpha: float = 0.01):
    for i, feature_name in enumerate(feature_names):
        if feature_name not in reference_stats:
            reference_stats[feature_name] = {
                'mean': 0.0,
                'std': 1.0,
                'min': float(feature_values[i]),
                'max': float(feature_values[i]),
                'count': 0
            }
        
        current_value = float(feature_values[i])
        
        if np.isnan(current_value) or np.isinf(current_value):
            continue
            
        stats = reference_stats[feature_name]
        
        if stats['count'] == 0:
            stats['mean'] = current_value
            stats['std'] = 0.0
        else:
            old_mean = stats['mean']
            stats['mean'] = (1 - alpha) * old_mean + alpha * current_value
            
            diff = current_value - old_mean
            stats['std'] = np.sqrt((1 - alpha) * stats['std']**2 + alpha * diff**2)
        
        stats['min'] = min(stats['min'], current_value)
        stats['max'] = max(stats['max'], current_value)
        stats['count'] += 1

def calculate_reference_statistics(X_train: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    reference_stats = {}
    for i, feature_name in enumerate(feature_names):
        feature_data = X_train[:, i]
        
        valid_data = feature_data[~(np.isnan(feature_data) | np.isinf(feature_data))]
        
        if len(valid_data) == 0:
            reference_stats[feature_name] = {
                'mean': 0.0,
                'std': 1.0,
                'min': 0.0,
                'max': 1.0,
                'count': 0
            }
        else:
            reference_stats[feature_name] = {
                'mean': float(np.mean(valid_data)),
                'std': max(float(np.std(valid_data)), 1e-6),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'count': len(valid_data)
            }
    
    return reference_stats