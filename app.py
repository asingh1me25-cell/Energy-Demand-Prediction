import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="1-Month Demand Predictor", layout="wide")

# ------------------ Load Models ------------------
@st.cache_resource
def load_models():
    xgb = pickle.load(open("models/vmd_xgb_model.pkl", "rb"))
    lstm = tf.keras.models.load_model("models/vmd_lstm_model.h5")
    fscaler = pickle.load(open("models/feature_scaler.pkl", "rb"))
    tscaler = pickle.load(open("models/target_scaler.pkl", "rb"))
    blend = pickle.load(open("models/blend_weight.pkl", "rb"))
    return xgb, lstm, fscaler, tscaler, blend

xgb_model, lstm_model, feature_scaler, target_scaler, blend_weight = load_models()

# ------------------ UI ------------------
st.title("⚡ 30-Day Electricity Demand Predictor (India Regions)")

col1, col2 = st.columns(2)

with col1:
    hour = st.number_input("Select Hour (0–23)", min_value=0, max_value=23, value=14)

with col2:
    region = st.selectbox(
        "Select Region",
        ["South", "North", "East", "West", "Total (All Regions)"]
    )

predict_btn = st.button("Predict Next 30 Days")

# ------------------ Prediction Logic ------------------
def predict_region(hour, region):
    X = np.array([[hour]] * 30)
    X_scaled = feature_scaler.transform(X)

    pred_xgb = xgb_model.predict(X_scaled)
    pred_lstm = lstm_model.predict(X_scaled).flatten()

    w = float(blend_weight)
    blended = (w * pred_lstm) + ((1 - w) * pred_xgb)

    final = target_scaler.inverse_transform(blended.reshape(-1, 1)).flatten()

    today = datetime.now()

    df = pd.DataFrame({
        "date": [(today + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(30)],
        "demand": np.round(final, 0).astype(int)
    })

    return df

# ------------------ Total Region Calculation ------------------
def predict_total(hour):
    regions = ["South", "North", "East", "West"]
    dfs = [predict_region(hour, r) for r in regions]

    total_df = dfs[0].copy()
    total_df["demand"] = sum(df["demand"] for df in dfs)

    return total_df

# ------------------ On Predict ------------------
if predict_btn:
    if region == "Total (All Regions)":
        df_pred = predict_total(hour)
    else:
        df_pred = predict_region(hour, region)

    st.subheader(f"📈 30-Day Forecast — {region}")

    st.dataframe(df_pred)

    # Chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_pred["date"], df_pred["demand"], marker="o")
    ax.set_xticks(df_pred["date"][::3])
    plt.xticks(rotation=45)
    ax.set_title("30-Day Demand Forecast")
    ax.set_ylabel("Demand (MW)")
    st.pyplot(fig)

    # Download
    st.download_button(
        "Download CSV",
        df_pred.to_csv(index=False),
        file_name=f"{region}_30day_forecast.csv",
        mime="text/csv"
    )
