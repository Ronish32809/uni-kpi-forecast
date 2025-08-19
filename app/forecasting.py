
# Small library the Streamlit UI imports

import os, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---------- paths & labels ----------
DATA_ROOT = "data"   # repo has data/<uni-folder>/...
PATHS = {
    "lamar": {
        "Enrollment":           "lamar/lamar_cleaned_enrollment.csv",
        "Enrollment Breakdown": "lamar/lamar_enrollment_counts.csv",
        "Employees":            "lamar/lamar_total_employee.csv",
        "Employee Breakdown":   "lamar/lamar_employee_counts.csv",
        "R&D Expenditure":      "lamar/lamar_rd_expenditure.csv",
        "Total Graduates":      "lamar/lamar_total_grads.csv",
    },
    "eou": {
        "Enrollment":           "eastern_oregon/east_oregon_cleaned_enrollment.csv",
        "Enrollment Breakdown": "eastern_oregon/east_oregon_enrollment_counts.csv",
        "Employees":            "eastern_oregon/east_oregon_total_employee.csv",
        "Employee Breakdown":   "eastern_oregon/east_oregon_employee_counts.csv",
        "R&D Expenditure":      "eastern_oregon/eou_rd_expenditure.csv",
        "Total Graduates":      "eastern_oregon/eou_total_grads.csv",
    },
    "uab": {
        "Enrollment":           "uab/alabama_cleaned_enrollment.csv",
        "Enrollment Breakdown": "uab/alabama_enrollment_counts.csv",
        "Employees":            "uab/alabama_total_employee.csv",
        "Employee Breakdown":   "uab/alabama_employee_counts.csv",
        "R&D Expenditure":      "uab/uab_rd_expenditure.csv",
        "Total Graduates":      "uab/uab_total_grads.csv",
    },
}
RENAME = {
    "R&D Expenditure": {"R&D Expenditure ($M)": "R&D_Expenditure"},
    "Total Graduates": {"Total_Graduates": "Grad_Total"},
}
ENROLL_MAP = {
    "Undergrad_Total": "Undergraduate Students",
    "Grad_Total":      "Graduate Students",
    "Total_Male":      "Male Students",
    "Total_Female":    "Female Students",
}
EMP_MAP = {
    "Full_Time":       "Full-Time Employees",
    "Part_Time":       "Part-Time Employees",
    "Male_Employee":   "Male Employees",
    "Female_Employee": "Female Employees",
}

def csv_path(university: str, kpi: str) -> str:
    """Full path to the CSV for a given university + KPI."""
    try:
        rel = PATHS[university][kpi]
    except KeyError as e:
        raise KeyError(f"Unknown key: {e}. University={university}, KPI={kpi}") from e
    return os.path.join(DATA_ROOT, rel)

# ---------- data helpers ----------
def load_ts(path: str, date_col: str) -> pd.DataFrame:
    """Read CSV, coerce dates. If only a year is present, assume Jan 1 of that year."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}\nCWD={os.getcwd()}")
    df = pd.read_csv(path)
    if date_col not in df.columns:
        alt = "Year" if date_col == "Date" else "Date"
        if alt in df.columns:
            date_col = alt
        else:
            raise KeyError(f"Neither '{date_col}' nor '{alt}' found in {path}")
    df[date_col] = pd.to_datetime(
        df[date_col].apply(lambda x: f"{x}-01-01" if re.fullmatch(r"\d{4}", str(x).strip()) else x),
        errors="coerce",
    )
    return df.dropna(subset=[date_col]).set_index(date_col).sort_index()

def make_windows(arr: np.ndarray, steps: int):
    X, y = [], []
    for i in range(len(arr) - steps):
        X.append(arr[i:i+steps]); y.append(arr[i+steps])
    return np.array(X).reshape(-1, steps, 1), np.array(y).reshape(-1, 1)

def make_lags(s: pd.Series, steps: int) -> pd.DataFrame:
    out = pd.DataFrame({"y": s})
    for k in range(1, steps+1):
        out[f"lag_{k}"] = s.shift(k)
    return out.dropna()

# ---------- model ----------
def forecast_series(
    series: pd.Series,
    years: int,
    rf_lags: int,
    lstm_units: int,
    lstm_activation: str,
    rf_estimators: int,
):
    """
    Train LSTM + RF on a single time series and forecast `years` steps.
    Returns (future_index, lstm_pred, rf_pred).
    """
    s = series.astype(float).interpolate("linear")

    # LSTM
    sc = MinMaxScaler()
    scaled = sc.fit_transform(s.values.reshape(-1,1))
    X, y = make_windows(scaled.flatten(), rf_lags)
    if len(X) == 0:
        lstm_pred = np.array([float(s.iloc[-1])] * years, dtype=float)
    else:
        mdl = Sequential([LSTM(lstm_units, activation=lstm_activation, input_shape=(rf_lags,1)), Dense(1)])
        mdl.compile(optimizer=Adam(0.003), loss="mse")
        mdl.fit(X, y, epochs=200, verbose=0, validation_split=.20,
                callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
        seq, out = scaled[-rf_lags:].reshape(1, rf_lags, 1), []
        for _ in range(years):
            nxt = float(mdl.predict(seq, verbose=0)[0,0])
            out.append(nxt)
            seq = np.append(seq.flatten()[1:], nxt).reshape(1, rf_lags, 1)
        lstm_pred = sc.inverse_transform(np.array(out).reshape(-1,1)).ravel()

    # RF
    rf = RandomForestRegressor(n_estimators=rf_estimators, random_state=42, min_samples_leaf=2)
    lag_df = make_lags(s, rf_lags)
    if lag_df.empty:
        rf_pred = np.array([float(s.iloc[-1])] * years, dtype=float)
    else:
        rf.fit(lag_df.drop(columns="y"), lag_df["y"])
        vals, rf_pred = s.iloc[-rf_lags:].tolist(), []
        for _ in range(years):
            p = float(rf.predict(np.array(vals[-rf_lags:]).reshape(1,-1))[0])
            rf_pred.append(p); vals.append(p)
        rf_pred = np.array(rf_pred, dtype=float)

    start_year = int(s.index.year.max()) + 1
    future = pd.date_range(start=f"{start_year}", periods=years, freq="YS")
    return future, lstm_pred, rf_pred
