# utils/s3_download.py

import os
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def download_file_from_s3(s3_key: str, local_path: str):
    """Downloads a file from S3 to the specified local path."""
    try:
        print(f"Downloading `{s3_key}` from S3 bucket `{S3_BUCKET_NAME}` to `{local_path}`.")
        s3 = boto3.client('s3', region_name=AWS_REGION, aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
        print("Download successful.")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
