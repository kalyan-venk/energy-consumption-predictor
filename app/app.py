# app/app.py
import pandas as pd
from pathlib import Path
import joblib
import streamlit as st
from prophet.plot import plot_plotly
from prophet.serialize import model_to_json, model_from_json

DATA = Path(__file__).resolve().parents[1] / "data" / "energy_hourly.csv"
MODEL = Path(__file__).resolve().parents[1] / "models" / "prophet_energy.joblib"

st.set_page_config(page_title="Energy Consumption Predictor", layout="wide")
st.title("⚡ Energy Consumption Predictor (Prophet)")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")
    df = df.rename(columns={"timestamp": "ds", "consumption_kwh": "y"})
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL)

df = load_data()
m = load_model()

st.subheader("Historical consumption")
st.line_chart(df.set_index("ds")["y"])

horizon_hours = st.slider("Forecast horizon (hours)", 6, 72, 24, step=6)

# Generate future dataframe starting just after last historical point
last_date = df["ds"].max()
future = m.make_future_dataframe(periods=horizon_hours, freq="H", include_history=False)
forecast = m.predict(future)

st.subheader("Forecast (Future Only)")

future = m.make_future_dataframe(periods=horizon_hours, freq="H", include_history=False)

if forecast.empty:
    st.warning("Forecast data not generated. Try re-running the model training step.")
else:
    st.line_chart(forecast.set_index("ds")[["yhat"]])
    st.caption("Predicted energy consumption for upcoming hours.")

st.subheader("Recent performance (last 7 days)")

# Last 7 days of actual data
recent_hist = df.set_index("ds").iloc[-24*7:]

# Forecast data (index by datetime)
forecast_indexed = forecast.set_index("ds")

# Find overlapping timestamps (some models predict only future)
common_index = recent_hist.index.intersection(forecast_indexed.index)

if len(common_index) > 0:
    # If overlap exists, show comparison chart
    comp = pd.DataFrame({
        "actual": recent_hist.loc[common_index, "y"],
        "pred": forecast_indexed.loc[common_index, "yhat"]
    })
    st.line_chart(comp)
else:
    # If no overlap, show a friendly info message
    st.info("ℹ️ Forecast does not overlap with the last 7 days of actual data yet. "
            "This is expected because Prophet only predicts future timestamps.")