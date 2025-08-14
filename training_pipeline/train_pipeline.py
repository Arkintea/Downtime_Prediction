# training_pipeline/train_pipeline.py

import os
import pandas as pd
from dotenv import load_dotenv
import yaml
import mlflow
from data_preprocessing.initial_cleaning import split_and_save_data
from data_preprocessing.preprocessor import Preprocessor
from utils.s3_upload import upload_file_to_s3
from utils.s3_download import download_file_from_s3
from utils.save_encoder import save_encoder
from training_pipeline.train_model import train_model

load_dotenv()
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

with open("config/config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)

RAW_S3_KEY = CONFIG["paths"]["s3_key"]
LOCAL_RAW_FILE = CONFIG["paths"]["raw_data"]
TRAIN_FILE = CONFIG["paths"]["train_data"]
INFER_FILE = CONFIG["paths"]["inference_data"]
ENCODER_DIR = CONFIG["paths"]["encoder_dir"]

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("machine_downtime_training_pipeline")

def main():
    with mlflow.start_run(run_name="full_pipeline"):
        print("Downloading raw file from S3...")
        try:
            download_file_from_s3(RAW_S3_KEY, LOCAL_RAW_FILE)
            mlflow.log_param("data_source", "s3")
        except Exception as e:
            print(f"S3 download failed: {e}, checking for local file...")
            if not os.path.exists(LOCAL_RAW_FILE):
                raise FileNotFoundError(f"Neither S3 nor local file available: {LOCAL_RAW_FILE}")
            mlflow.log_param("data_source", "local")

        print("Loading raw data from local...")
        df = pd.read_excel(LOCAL_RAW_FILE)
        
        initial_rows = len(df)
        initial_cols = len(df.columns)
        mlflow.log_param("initial_rows", initial_rows)
        mlflow.log_param("initial_columns", initial_cols)

        print("Splitting data 85/15...")
        split_and_save_data(df, TRAIN_FILE, INFER_FILE)

        print("Uploading splits to S3...")
        try:
            upload_file_to_s3(TRAIN_FILE, "training_data/training_data.csv", BUCKET_NAME)
            upload_file_to_s3(INFER_FILE, "inference_data/inference_data.csv", BUCKET_NAME)
            mlflow.log_param("s3_upload_success", True)
        except Exception as e:
            print(f"S3 upload failed: {e}")
            mlflow.log_param("s3_upload_success", False)

        print("Downloading training split again for reproducibility...")
        try:
            download_file_from_s3("training_data/training_data.csv", TRAIN_FILE)
        except Exception as e:
            print(f"Using local training file: {e}")

        print("Loading training data...")
        df = pd.read_csv(TRAIN_FILE)
        
        train_rows = len(df)
        mlflow.log_param("train_rows_after_split", train_rows)

        print("Reconstructing datetime...")
        df = Preprocessor.reconstruct_datetime(df)

        print("Adding datetime features...")
        df = Preprocessor.add_datetime_features(df)

        print("Mapping downtime groups...")
        df = Preprocessor.map_downtime_groups(df)
        
        if 'Downtime_Group' in df.columns:
            downtime_distribution = df['Downtime_Group'].value_counts().to_dict()
            for group, count in downtime_distribution.items():
                mlflow.log_param(f"downtime_group_{group}", count)

        print("Label encoding downtime groups and saving encoder...")
        df, downtime_encoder = Preprocessor.label_encode_target(
            df, target_col="Downtime_Group", save_path=None
        )
        save_encoder({'Downtime_Group': downtime_encoder}, os.path.join(ENCODER_DIR, "label_encoder.pkl"))

        print("Creating future downtime target for T-10 prediction...")
        df = Preprocessor.create_future_target(df, target_col="Downtime_Group", shift_periods=10)
        
        final_rows_after_shift = len(df)
        mlflow.log_param("final_rows_after_target_shift", final_rows_after_shift)

        print("Adding temporal features (lag, rolling mean/std)...")
        df = Preprocessor.add_temporal_features(df)
        
        final_rows_after_temporal = len(df)
        mlflow.log_param("final_rows_after_temporal_features", final_rows_after_temporal)

        print("One-hot encoding categorical features...")
        df = Preprocessor.one_hot_encode_features(
            df,
            columns=['Machine_ID', 'Assembly_Line_No'],
            save_path=os.path.join(ENCODER_DIR, "onehot_columns.pkl")
        )
        
        final_columns = len(df.columns)
        mlflow.log_param("final_columns_after_encoding", final_columns)

        print("Dropping unused columns...")
        df = Preprocessor.drop_unused_columns(df, ['Downtime', 'Downtime_Group'])
        
        if 'Future_Downtime_Label' in df.columns:
            target_distribution = df['Future_Downtime_Label'].value_counts().to_dict()
            for label, count in target_distribution.items():
                mlflow.log_param(f"target_class_{label}", count)

        print("Training models for all machine-line sections...")
        training_results = train_model(df)
        
        # Log final results
        mlflow.log_param("best_hyperparameters", str(training_results['best_params']))
        mlflow.log_metric("final_test_recall", training_results['test_recall'])
        mlflow.log_metric("final_cv_recall", training_results['cv_recall'])
        
        # Log model file paths for reference
        mlflow.log_param("production_model_path", training_results.get('model_path', 'artifacts/models/model.pkl'))
        mlflow.log_param("production_scaler_path", training_results.get('scaler_path', 'artifacts/models/scaler.pkl'))

        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETE")
        print("="*60)
        print(f"✅ Final test recall score: {training_results['test_recall']:.4f}")
        print(f"✅ Best CV recall score: {training_results['cv_recall']:.4f}")
        print(f"✅ Best model saved locally: {training_results.get('model_path', 'artifacts/models/model.pkl')}")
        print(f"✅ Scaler saved locally: {training_results.get('scaler_path', 'artifacts/models/scaler.pkl')}")
        print("="*60)
        
        return training_results

if __name__ == "__main__":
    main()