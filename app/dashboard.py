import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from app.forecasting import (
    PATHS, RENAME, ENROLL_MAP, EMP_MAP,
    load_ts, run_forecast
)

st.set_page_config(page_title="University KPI Forecasts", layout="wide")
st.title("University KPI Forecast Dashboard")
st.caption("Interactive forecasts using LSTM + Random Forest")

with st.sidebar:
    university = st.selectbox("University", ["lamar", "eou", "uab"], index=0)
    kpi_to_run = st.selectbox(
        "KPI",
        ["Enrollment", "Enrollment Breakdown", "Employees",
         "Employee Breakdown", "R&D Expenditure", "Total Graduates"],
        index=4
    )
    forecast_years = st.slider("Forecast years", 1, 10, 5)
    lstm_units = st.select_slider("LSTM units",
                                  options=[8,16,24,32,40,48,56,64,96,128],
                                  value=32)
    lstm_activation = st.selectbox("LSTM activation", ["tanh","relu","sigmoid"], index=0)
    rf_lags = st.slider("RF/LSTM lookback (lags)", 1, 10, 4)

# pick CSV and load
csv = PATHS(university)[kpi_to_run]
time_col = "Year" if kpi_to_run in ["R&D Expenditure", "Total Graduates"] else "Date"
df = load_ts(csv, time_col).rename(columns=RENAME.get(kpi_to_run, {}))

def plot_and_table(df, col, title):
    future, lstm_pred, rf_pred = run_forecast(
        df, col, title,
        forecast_years=forecast_years,
        rf_lags=rf_lags,
        lstm_units=lstm_units,
        lstm_activation=lstm_activation
    )
    lstm_df = pd.DataFrame(lstm_pred, index=future, columns=["LSTM"])
    rf_df   = pd.DataFrame(rf_pred,   index=future, columns=["RF"])

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df.index, df[col], "o-", label="Actual")
    ax.plot(lstm_df.index, lstm_df["LSTM"], "--o", label="LSTM")
    ax.plot(rf_df.index, rf_df["RF"], "--x", label="RF")
    ax.set_title(f"{title}: {future.year[0]}–{future.year[-1]}")
    ax.grid(); ax.legend()
    st.pyplot(fig)
    st.dataframe(pd.concat([lstm_df, rf_df], axis=1))

# branch per KPI
if kpi_to_run == "Enrollment":
    df["Total"] = df.sum(axis=1)
    plot_and_table(df, "Total", f"{university.upper()} – Total Enrollment")
elif kpi_to_run == "Enrollment Breakdown":
    for c,lbl in ENROLL_MAP.items():
        if c in df.columns:
            plot_and_table(df[[c]], c, f"{university.upper()} – {lbl}")
elif kpi_to_run == "Employees":
    df["Total"] = df.sum(axis=1)
    plot_and_table(df, "Total", f"{university.upper()} – Total Employees")
elif kpi_to_run == "Employee Breakdown":
    for c,lbl in EMP_MAP.items():
        if c in df.columns:
            plot_and_table(df[[c]], c, f"{university.upper()} – {lbl}")
elif kpi_to_run == "R&D Expenditure":
    plot_and_table(df, "R&D_Expenditure", f"{university.upper()} – R&D Expenditure")
elif kpi_to_run == "Total Graduates":
    grad_col = next(col for col in df.columns if "Grad" in col or "Total" in col)
    plot_and_table(df, grad_col, f"{university.upper()} – Total Graduates")
