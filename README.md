# Readme

## Setup
./venv/Scripts/activate

pip install -r requirements.txt

## Running Inference

pip install uvicorn fastapi

python streaming/publish_to_mqtt.py

python streaming/consume_kafka_and_infer.py

uvicorn inference.main:app --reload

streamlit run dashboard/streamlit_dashboard.py


## Training
python training_pipeline/train_pipeline.py


## Production

docker build -t Downtime-Prediction .

docker run -d -p 8000:8000 Downtime-Prediction

