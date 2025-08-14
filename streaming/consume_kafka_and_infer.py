# streaming/consume_kafka_and_infer.py
import os
import json
import time
import requests
from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv

load_dotenv()

conf = {
    'bootstrap.servers': os.getenv("BOOTSTRAP_SERVERS"),
    'security.protocol': 'SASL_SSL',
    'sasl.mechanism': 'PLAIN',
    'sasl.username': os.getenv("KAFKA_API_KEY"),
    'sasl.password': os.getenv("KAFKA_API_SECRET"),
    'group.id': 'inference-consumer-group',
    'auto.offset.reset': 'earliest'
}

topic = os.getenv("KAFKA_TOPIC")
fastapi_url = os.getenv("FASTAPI_ENDPOINT")

consumer = Consumer(conf)
consumer.subscribe([topic])

print(f"Listening to Kafka topic: {topic}")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print(f"Kafka error: {msg.error()}")
            continue

        try:
            payload = json.loads(msg.value().decode("utf-8"))

            request_body = {
                "machine_id": payload.get("Machine_ID"),
                "assembly_line_no": payload.get("Assembly_Line_No"),
                "data": payload }

            print(f"Received message: {payload}")
            response = requests.post(fastapi_url, json=request_body)
            if response.status_code == 200:
                print(f"Inference result: {response.json()}")
            else:
                print(f"Failed inference: {response.status_code} → {response.text}")
        except Exception as e:
            print(f"Processing error: {e}")

except KeyboardInterrupt:
    pass
finally:
    consumer.close()
