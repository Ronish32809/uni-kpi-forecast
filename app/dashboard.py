# app/dashboard.py
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from forecasting import (
    PATHS, RENAME, ENROLL_MAP, EMP_MAP,
    csv_path, load_ts, forecast_series
)

st.set_page_config(page_title="University KPI Forecasts", layout="wide")
st.title("University KPI Forecasts")

# ---------- Sidebar controls ----------
st.sidebar.header("Controls")
uni = st.sidebar.selectbox("University", ["lamar", "eou", "uab", "all"], index=0)
kpi = st.sidebar.selectbox(
    "KPI",
    ["Enrollment", "Enrollment Breakdown", "Employees", "Employee Breakdown",
     "R&D Expenditure", "Total Graduates"],
    index=4
)
years = st.sidebar.slider("Forecast years", 1, 10, 5)

with st.sidebar.expander("Model settings", expanded=False):
    lstm_units = st.slider("LSTM units", 8, 128, 32, step=8)
    lstm_activation = st.selectbox("LSTM activation", ["tanh", "relu", "sigmoid"], index=0)
    rf_lags = st.slider("RF lags (lookback)", 1, 10, 4, step=1)
    rf_estimators = st.slider("RF trees", 10, 500, 200, step=10)

run = st.sidebar.button("Run forecast")

# ---------- Helpers ----------
def counts_like_title(title: str) -> bool:
    return any(t in title for t in ["Enrollment", "Employee", "Graduate"])

def plot_series(actual_idx, actual_vals, future_idx, lstm_pred, rf_pred, title):
    # round counts; keep money to 2dp
    if counts_like_title(title):
        lstm_pred = np.round(lstm_pred).astype(int)
        rf_pred   = np.round(rf_pred).astype(int)
    else:
        lstm_pred = np.round(lstm_pred, 2)
        rf_pred   = np.round(rf_pred, 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual_idx, actual_vals, "o-", label="Actual")
    ax.plot(future_idx, lstm_pred, "--o", label="LSTM")
    ax.plot(future_idx, rf_pred,  "--x", label="RF")
    ax.set_title(title)
    ax.grid(True); ax.legend()

    table = pd.concat(
        [pd.DataFrame(lstm_pred, index=future_idx, columns=["LSTM"]),
         pd.DataFrame(rf_pred,   index=future_idx, columns=["RF"])],
        axis=1
    )
    return fig, table

@st.cache_data(show_spinner=False)
def load_df(university: str, kpi_name: str) -> pd.DataFrame:
    """Cached CSV load + rename using the shared forecasting helpers."""
    # Decide date column by KPI (yearly vs dated)
    date_col = "Year" if kpi_name in ["R&D Expenditure", "Total Graduates"] else "Date"
    csv = csv_path(university, kpi_name)
    if not os.path.exists(csv):
        raise FileNotFoundError(f"Data file not found: {csv}\nCWD={os.getcwd()}")
    df = load_ts(csv, date_col).rename(columns=RENAME.get(kpi_name, {}))
    return df

# ---------- App logic ----------
if run:
    try:
        # SINGLE UNIVERSITY
        if uni != "all":
            df = load_df(uni, kpi)

            # totals vs breakdowns
            if kpi == "Enrollment":
                series = df.sum(axis=1)
                title  = f"{uni.upper()} – Total Enrollment"
            elif kpi == "Employees":
                series = df.sum(axis=1)
                title  = f"{uni.upper()} – Total Employees"
            elif kpi == "R&D Expenditure":
                series = df["R&D_Expenditure"]
                title  = f"{uni.upper()} – R&D Expenditure"
            elif kpi == "Total Graduates":
                col = next((c for c in df.columns if "Grad" in c or "Total" in c), df.columns[0])
                series = df[col]
                title  = f"{uni.upper()} – Total Graduates"
            elif kpi == "Enrollment Breakdown":
                for col, lbl in ENROLL_MAP.items():
                    if col in df.columns:
                        s = df[col]
                        future, l_pred, r_pred = forecast_series(
                            s, years, rf_lags, lstm_units, lstm_activation, rf_estimators
                        )
                        fig, table = plot_series(s.index, s.values, future, l_pred, r_pred,
                                                 f"{uni.upper()} – {lbl}")
                        st.subheader(lbl)
                        st.pyplot(fig, clear_figure=True)
                        st.dataframe(table)
                st.stop()
            else:  # Employee Breakdown
                for col, lbl in EMP_MAP.items():
                    if col in df.columns:
                        s = df[col]
                        future, l_pred, r_pred = forecast_series(
                            s, years, rf_lags, lstm_units, lstm_activation, rf_estimators
                        )
                        fig, table = plot_series(s.index, s.values, future, l_pred, r_pred,
                                                 f"{uni.upper()} – {lbl}")
                        st.subheader(lbl)
                        st.pyplot(fig, clear_figure=True)
                        st.dataframe(table)
                st.stop()

            # run forecast for the chosen single-series
            future, l_pred, r_pred = forecast_series(
                series, years, rf_lags, lstm_units, lstm_activation, rf_estimators
            )
            fig, table = plot_series(series.index, series.values, future, l_pred, r_pred, title)
            st.pyplot(fig, clear_figure=True)
            st.dataframe(table)

        # ALL UNIVERSITIES (overlay)
        else:
            schools = ["lamar", "eou", "uab"]

            if kpi in ["Enrollment Breakdown", "Employee Breakdown"]:
                metrics = ENROLL_MAP if kpi == "Enrollment Breakdown" else EMP_MAP
                for mcol, mlbl in metrics.items():
                    figs_tables = []
                    fig, ax = plt.subplots(figsize=(12, 5))
                    out_cols = []

                    for s in schools:
                        df = load_df(s, kpi)
                        if mcol not in df.columns:  # skip missing series
                            continue
                        series = df[mcol]
                        future, l_pred, r_pred = forecast_series(
                            series, years, rf_lags, lstm_units, lstm_activation, rf_estimators
                        )
                        # overlay actual + preds
                        ax.plot(series.index, series.values, "o-", label=f"{s.upper()} Actual")
                        ax.plot(future, np.round(l_pred).astype(int), "--o", label=f"{s.upper()} LSTM")
                        ax.plot(future, np.round(r_pred).astype(int), "--x", label=f"{s.upper()} RF")
                        # table columns per school
                        out_cols.append(pd.concat({
                            f"{s.upper()} LSTM": pd.Series(np.round(l_pred).astype(int), index=future),
                            f"{s.upper()} RF":   pd.Series(np.round(r_pred).astype(int), index=future)
                        }, axis=1))

                    ax.set_title(f"{mlbl} – All Universities")
                    ax.grid(True); ax.legend()
                    st.subheader(mlbl)
                    st.pyplot(fig, clear_figure=True)
                    if out_cols:
                        st.dataframe(pd.concat(out_cols, axis=1))

            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                out_cols = []

                for s in schools:
                    df = load_df(s, kpi)

                    if kpi in ["Enrollment", "Employees"]:
                        series = df.sum(axis=1); counts_like = True
                    elif kpi == "R&D Expenditure":
                        series = df["R&D_Expenditure"]; counts_like = False
                    else:  # Total Graduates
                        col = next((c for c in df.columns if "Grad" in c or "Total" in c), df.columns[0])
                        series = df[col]; counts_like = True

                    future, l_pred, r_pred = forecast_series(
                        series, years, rf_lags, lstm_units, lstm_activation, rf_estimators
                    )

                    # formatting per type
                    if counts_like:
                        l_plot = np.round(l_pred).astype(int); r_plot = np.round(r_pred).astype(int)
                    else:
                        l_plot = np.round(l_pred, 2);         r_plot = np.round(r_pred, 2)

                    ax.plot(series.index, series.values, "o-", label=f"{s.upper()} Actual")
                    ax.plot(future, l_plot, "--o", label=f"{s.upper()} LSTM")
                    ax.plot(future, r_plot, "--x", label=f"{s.upper()} RF")

                    out_cols.append(pd.concat({
                        f"{s.upper()} LSTM": pd.Series(l_plot, index=future),
                        f"{s.upper()} RF":   pd.Series(r_plot, index=future)
                    }, axis=1))

                ax.set_title(f"{kpi} – All Universities")
                ax.grid(True); ax.legend()
                st.pyplot(fig, clear_figure=True)
                if out_cols:
                    st.dataframe(pd.concat(out_cols, axis=1))

    except Exception as e:
        st.error(str(e))
        st.stop()

st.caption("Tip: run from the repo root so relative paths resolve (folder with /app and /data).")
