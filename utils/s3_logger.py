# utils/s3_logger.py
import os
import boto3
import pandas as pd
import tempfile
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def push_log_to_s3(df: pd.DataFrame, prefix: str = "latest_predictions") -> bool:
    try:
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, BUCKET_NAME]):
            print("Missing AWS credentials or S3 bucket name in environment variables")
            return False
        s3 = boto3.client('s3', region_name=AWS_REGION, aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        filename = f"{prefix}/predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        
        try:
            s3.upload_file(tmp_file_path, BUCKET_NAME, filename)
            print(f"✅ Uploaded prediction log to s3://{BUCKET_NAME}/{filename}")
            return True
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return False