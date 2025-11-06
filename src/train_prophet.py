# src/train_prophet.py
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from pathlib import Path
import joblib

DATA = Path(__file__).resolve().parents[1] / "data" / "energy_hourly.csv"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.rename(columns={"timestamp": "ds", "consumption_kwh": "y"})
    return df

def train_and_eval(df):
    # Hold out last 30 days for evaluation
    cutoff = df["ds"].max() - pd.Timedelta(days=30)
    train = df[df["ds"] <= cutoff].copy()
    test  = df[df["ds"]  > cutoff].copy()

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    m.fit(train)

    future = m.make_future_dataframe(periods=24, freq="H")  # next 24 hours
    forecast = m.predict(future)

    # Evaluate on the test period (Prophet produces fitted values for history)
    # Align test timestamps with available forecasts
    forecast_df = forecast[["ds", "yhat"]].set_index("ds")
    aligned = test.set_index("ds").join(forecast_df, how="inner")
    if aligned.empty:
        print("Warning: No overlapping timestamps for evaluation. Skipping MAPE.")
        mape = None
    else:
        mape = mean_absolute_percentage_error(aligned["y"], aligned["yhat"])

    print(f"Test MAPE over last 30 days: {mape if mape else 'N/A'}")

    # Save
    joblib.dump(m, MODEL_DIR / "prophet_energy.joblib")

    print(f"Test MAPE over last 30 days: {mape:.4f}")
    return mape

if __name__ == "__main__":
    df = load_data()
    mape = train_and_eval(df)