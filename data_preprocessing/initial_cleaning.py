#data_preprocessing/initial_cleaning.py

import os
import pandas as pd
from dotenv import load_dotenv
from utils.s3_upload import upload_file_to_s3

load_dotenv()
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def split_and_save_data(df, train_output_path, infer_output_path):
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(infer_output_path), exist_ok=True)
    
    split_index = int(len(df) * 0.85)
    df_train = df.iloc[:split_index]
    df_infer = df.iloc[split_index:]

    # Save splits locally
    print(f"✅Saving training split to: {train_output_path}")
    df_train.to_csv(train_output_path, index=True)

    print(f"✅Saving inference split to: {infer_output_path}")
    df_infer.to_csv(infer_output_path, index=True)

    # Upload to S3
    print("✅Uploading training data to S3...")
    upload_file_to_s3(train_output_path, "training_data/training_data.csv", BUCKET_NAME)

    print("✅Uploading inference data to S3...")
    upload_file_to_s3(infer_output_path, "inference_data/inference_data.csv", BUCKET_NAME)

    print("✅Split and upload complete.")
    return df_train, df_infer
