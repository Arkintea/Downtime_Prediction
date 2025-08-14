# dashboard/streamlit_dashboard.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import yaml

try:
    with open("config/config.yaml", "r") as file:
        CONFIG = yaml.safe_load(file)
    REFRESH_INTERVAL = CONFIG["dashboard"]["refresh_interval"]
except FileNotFoundError:
    REFRESH_INTERVAL = 20000

FASTAPI_URL = "http://localhost:8000"

st.set_page_config(page_title="Downtime Prediction Dashboard", layout="wide")
st.title("🔧 Machine Downtime Predictions")

st_autorefresh(interval=REFRESH_INTERVAL, limit=None, key="auto_refresh")

with st.sidebar:
    st.header("⚙️ System Status")
    try:
        health_response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health = health_response.json()
            st.success("API: ✅ Online")
            st.success("Model: ✅ Loaded")
            st.info("Monitoring: ✅ Active")
        else:
            st.error("API: ❌ Error")
    except requests.exceptions.RequestException:
        st.error("API: ❌ Offline")

if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

@st.cache_data(ttl=10)
def fetch_latest_prediction():
    try:
        response = requests.get(f"{FASTAPI_URL}/latest_predictions", timeout=5)
        if response.status_code == 200:
            predictions = response.json().get("predictions", [])
            if predictions:
                return predictions[-1]
        return None
    except requests.exceptions.RequestException:
        return None

st.header("📡 Latest Prediction")

prediction = fetch_latest_prediction()

if prediction:
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Machine", prediction.get("machine_id", "N/A"))
        st.metric("Line", prediction.get("assembly_line_no", "N/A"))
    
    with col2:
        downtime = prediction.get("predicted_downtime", "N/A")
        if downtime == "Normal":
            st.metric("Predicted Downtime", f"🟢 {downtime}")
        elif downtime == "Infrastructure":
            st.metric("Predicted Downtime", f"🟡 {downtime}")
        elif downtime == "Machine_Issue":
            st.metric("Predicted Downtime", f"🔴 {downtime}")
        else:
            st.metric("Predicted Downtime", f"🔵 {downtime}")
        

    if not st.session_state.prediction_log or st.session_state.prediction_log[-1].get('timestamp') != prediction.get('timestamp'):
        st.session_state.prediction_log.append(prediction)
        if len(st.session_state.prediction_log) > 500:
            st.session_state.prediction_log = st.session_state.prediction_log[-500:]

else:
    st.info("Waiting for predictions...")

st.divider()
st.subheader("📜 Recent Predictions")

if st.session_state.prediction_log:
    display_data = []
    
    for pred in st.session_state.prediction_log[-20:]:
        timestamp = pred.get("timestamp", "")
        try:
            formatted_time = pd.to_datetime(timestamp).strftime("%H:%M:%S")
        except:
            formatted_time = timestamp
            
        downtime = pred.get("predicted_downtime", "N/A")
        
        status_icon = "🟢" if downtime == "Normal" else "🟡" if downtime == "Infrastructure" else "🔴" if downtime == "Machine_Issue" else "🔵"
        
        display_data.append({
            "Time": formatted_time,
            "Machine": pred.get("machine_id", ""),
            "Line": pred.get("assembly_line_no", ""),
            "Status": f"{status_icon} {downtime}",
        })
    
    if display_data:
        df_display = pd.DataFrame(display_data)
        df_display = df_display.iloc[::-1]
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions available")
else:
    st.info("No prediction history")