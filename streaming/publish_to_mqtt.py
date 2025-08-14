# streaming/publish_to_mqtt.py
import os
import time
import json
import ssl
import yaml
import pandas as pd
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Load secrets from .env
load_dotenv()
USERNAME = os.getenv("MQTT_USERNAME")
PASSWORD = os.getenv("MQTT_PASSWORD")

# Load CONFIG values from YAML
with open("config/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

BROKER = CONFIG["mqtt"]["broker"]
PORT = CONFIG["mqtt"]["port"]
TOPIC = CONFIG["mqtt"]["topic"]
SLEEP_INTERVAL = CONFIG["mqtt"]["sleep_interval"]
LOCAL_PATH = CONFIG["paths"]["inference_data"]

# Setup MQTT client
client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)
client.connect(BROKER, PORT)
client.loop_start()

def stream_inference_data_once():
    try:
        df = pd.read_csv(LOCAL_PATH)
        for idx, row in df.iterrows():
            message = row.drop("Downtime", errors='ignore').to_dict()
            client.publish(TOPIC, json.dumps(message))
            print(f"[{idx}] Published to MQTT: {message}")
            time.sleep(SLEEP_INTERVAL)
        print("Finished streaming all rows.")
    except FileNotFoundError:
        print(f"File not found: {LOCAL_PATH}")
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
if __name__ == "__main__":
    stream_inference_data_once()
