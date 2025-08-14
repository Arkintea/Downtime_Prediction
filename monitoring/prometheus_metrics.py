# monitoring/prometheus_metrics.py
import os
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import atexit

# Metrics for monitoring
prediction_counter = Counter('ml_predictions_total', 'Total number of predictions made')
drift_detected_counter = Counter('ml_drift_detected_total', 'Total number of drift detections')
anomaly_detected_counter = Counter('ml_anomalies_detected_total', 'Total number of anomalies detected')

prediction_confidence = Histogram('ml_prediction_confidence', 'Distribution of prediction confidence scores')
drift_features_count = Gauge('ml_drifted_features_count', 'Number of features showing drift')
model_health = Gauge('ml_model_health_status', 'Model health status (1=healthy, 0=unhealthy)')

# Feature-specific drift metrics
feature_drift_gauges = {}

def init_prometheus_server(port=8001):
    """Start Prometheus metrics server"""
    try:
        start_http_server(port)
        print(f"Prometheus metrics server started on port {port}")
        model_health.set(1)  # Set initial health status
    except Exception as e:
        print(f"Failed to start Prometheus server: {e}")

def record_prediction(confidence_score):
    """Record a prediction with its confidence"""
    prediction_counter.inc()
    prediction_confidence.observe(confidence_score)

def record_drift_detection(drifted_features_list):
    """Record drift detection event"""
    if drifted_features_list:
        drift_detected_counter.inc()
        drift_features_count.set(len(drifted_features_list))
        
        # Update individual feature drift status
        for feature in drifted_features_list:
            if feature not in feature_drift_gauges:
                feature_drift_gauges[feature] = Gauge(
                    f'ml_feature_drift_{feature.replace("(", "_").replace(")", "_").replace("?", "_")}', 
                    f'Drift status for feature {feature}'
                )
            feature_drift_gauges[feature].set(1)
    else:
        drift_features_count.set(0)

def record_anomaly_detection(is_anomaly):
    """Record anomaly detection result"""
    if is_anomaly:
        anomaly_detected_counter.inc()

def update_model_health(is_healthy):
    """Update model health status"""
    model_health.set(1 if is_healthy else 0)

def cleanup_prometheus():
    """Cleanup function for graceful shutdown"""
    model_health.set(0)

# Register cleanup function
atexit.register(cleanup_prometheus)