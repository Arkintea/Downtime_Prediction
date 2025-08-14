# training_pipeline/train_model.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, make_scorer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from monitoring.drift_detector import (
    initialise_drift_detector, 
    save_drift_detector, 
    calculate_reference_statistics
)
from monitoring.anomaly_detector import fit_mahalanobis
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

def train_model(df, model_dir="artifacts/models"):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("artifacts/drift", exist_ok=True)
    os.makedirs("artifacts/anomaly", exist_ok=True)
    os.makedirs("encoders", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    
    print("Creating train/test splits...")
    df_train, df_test = train_test_split(df, test_size=0.15, shuffle=False, stratify=None)

    X_train = df_train.drop('Future_Downtime_Label', axis=1)
    y_train = df_train['Future_Downtime_Label']
    X_test = df_test.drop('Future_Downtime_Label', axis=1)
    y_test = df_test['Future_Downtime_Label']

    feature_columns = X_train.columns.tolist()

    feature_names_path = "encoders/feature_names.pkl"
    joblib.dump(feature_columns, feature_names_path)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Initialising drift detection system...")
    drift_detectors, _ = initialise_drift_detector(feature_columns)
    
    sample_size = min(1000, len(X_train_scaled))
    sample_indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
    
    for idx in sample_indices:
        row = X_train_scaled[idx]
        for i, feature_name in enumerate(feature_columns):
            if feature_name in drift_detectors:
                drift_detectors[feature_name].update(float(row[i]))
    
    reference_stats = calculate_reference_statistics(X_train_scaled, feature_columns)
    save_drift_detector(drift_detectors, reference_stats)

    print("Fitting anomaly detection model...")
    try:
        fit_mahalanobis(X_train_scaled, contamination=0.01)
    except Exception as e:
        print(f"Failed to fit anomaly detector: {e}")

    print("Applying SMOTE for class balance...")
    try:
        min_samples = Counter(y_train).most_common()[-1][1]
        k_neighbors = min(5, max(1, min_samples - 1))
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    except Exception as e:
        print(f"SMOTE failed: {e}, using original data")
        X_train_resampled, y_train_resampled = X_train_scaled, y_train

    print("Training XGBoost with hyperparameter tuning...")
    
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [6, 10],
        'learning_rate': [0.1]
    }
    
    base_model = XGBClassifier(
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=0
    )
    
    recall_scorer = make_scorer(recall_score, average='macro', zero_division=0)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=recall_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_recall = grid_search.best_score_
    
    try:
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_recall_score", best_cv_recall)
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))
        
        cv_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=cv, scoring=recall_scorer)
        mlflow.log_metric("cv_recall_mean", cv_scores.mean())
        mlflow.log_metric("cv_recall_std", cv_scores.std())
    except Exception as e:
        print(f"MLflow logging error: {e}")

    print("Evaluating model performance...")
    y_pred = best_model.predict(X_test_scaled)
    test_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    try:
        mlflow.log_metric("test_recall_score", test_recall)
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"recall_{class_name}", metrics.get('recall', 0))
                mlflow.log_metric(f"precision_{class_name}", metrics.get('precision', 0))
                mlflow.log_metric(f"f1_{class_name}", metrics.get('f1-score', 0))
    except Exception as e:
        print(f"MLflow metric logging error: {e}")

    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    try:
        input_example = X_train_scaled[:1]
        
        mlflow.xgboost.log_model(
            best_model, 
            "best_model",
            input_example=input_example,
            signature=mlflow.models.infer_signature(input_example, best_model.predict(input_example))
        )
        
        mlflow.sklearn.log_model(
            scaler, 
            "scaler",
            input_example=X_train.values[:1],
            signature=mlflow.models.infer_signature(X_train.values[:1], scaler.transform(X_train.values[:1]))
        )
        
        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("local_model_path", model_path)
        
    except Exception as e:
        print(f"MLflow model logging error: {e}")

    return {
        'model': best_model,
        'scaler': scaler,
        'drift_detectors': drift_detectors,
        'reference_stats': reference_stats,
        'best_params': best_params,
        'test_recall': test_recall,
        'cv_recall': best_cv_recall,
        'report': report,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'model_path': model_path,
        'scaler_path': scaler_path
    }