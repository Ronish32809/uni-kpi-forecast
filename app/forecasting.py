# USER-CONFIGURABLE SETTINGS 
university = "lamar"  # @param ["lamar", "eou", "uab"]

kpi_to_run = "R&D Expenditure"  # @param ["Enrollment", "Enrollment Breakdown", "Employees", "Employee Breakdown", "R&D Expenditure", "Total Graduates"]

forecast_years = 5  # @param {type:"slider", min:1, max:10, step:1}

# LSTM hyper-params
lstm_units = 32  # @param {type:"slider", min:8, max:128, step:8}
lstm_activation = "tanh"  # @param ["tanh", "relu", "sigmoid"]

# Random-Forest hyper-params
rf_lags = 4  # @param {type:"slider", min:1, max:10, step:1}
rf_estimators = 200  # @param {type:"slider", min:10, max:500, step:10}


import re, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from IPython.display import display

# folder names
BASE = {"lamar": "lamar",
        "eou":   "eastern_oregon",
        "uab":   "uab"}[university]

# csv locations we actually have
PATHS = {
    "Enrollment":           f"{BASE}/{university}_cleaned_enrollment.csv",
    "Enrollment Breakdown": f"{BASE}/{university}_enrollment_counts.csv",
    "Employees":            f"{BASE}/{university}_total_employee.csv",
    "Employee Breakdown":   f"{BASE}/{university}_employee_counts.csv",
    "R&D Expenditure":      f"{BASE}/{university}_rd_expenditure.csv",
    "Total Graduates":      f"{BASE}/{university}_total_grads.csv", 
}

# fix headers once so the rest of the code can assume one name
RENAME = {
    "R&D Expenditure": {"R&D Expenditure ($M)": "R&D_Expenditure"},
    "Total Graduates": {"Total_Graduates": "Grad_Total"},
}

# labels for breakdown sheets
ENROLL_MAP = {"Undergrad_Total":"Undergraduate Students",
              "Grad_Total":"Graduate Students",
              "Total_Male":"Male Students",
              "Total_Female":"Female Students"}
EMP_MAP    = {"Full_Time":"Full-Time Employees",
              "Part_Time":"Part-Time Employees",
              "Male_Employee":"Male Employees",
              "Female_Employee":"Female Employees"}

# helper: load CSV and coerce dates
def load_ts(path, date_col):
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(
        df[date_col].apply(lambda x: f"{x}-01-01"
                           if re.fullmatch(r"\d{4}", str(x).strip()) else x),
        errors="coerce")
    return df.dropna(subset=[date_col]).set_index(date_col).sort_index()

#helper: windows + lags 
def make_windows(arr, steps):
    X, y = [], []
    for i in range(len(arr) - steps):
        X.append(arr[i:i+steps]); y.append(arr[i+steps])
    return np.array(X).reshape(-1, steps, 1), np.array(y).reshape(-1, 1)

def make_lags(s, steps):
    out = pd.DataFrame({"y": s})
    for k in range(1, steps+1): out[f"lag_{k}"] = s.shift(k)
    return out.dropna()

#  main model + plot 
def forecast(df, col, title):
    df[col] = df[col].interpolate("linear")

    # LSTM
    sc = MinMaxScaler()
    scaled = sc.fit_transform(df[[col]])
    X, y = make_windows(scaled.flatten(), rf_lags)

    lstm = Sequential([LSTM(lstm_units, activation=lstm_activation,
                            input_shape=(rf_lags,1)),
                       Dense(1)])
    lstm.compile(Adam(0.003), loss="mse")
    lstm.fit(X, y, 200, verbose=0, validation_split=.25,
             callbacks=[EarlyStopping("val_loss",10,restore_best_weights=True)])

    seq = scaled[-rf_lags:].reshape(1, rf_lags, 1)
    lstm_scaled = []
    for _ in range(forecast_years):
        nxt = lstm.predict(seq, verbose=0)[0,0]
        lstm_scaled.append(nxt)
        seq = np.append(seq.flatten()[1:], nxt).reshape(1, rf_lags, 1)
    lstm_pred = sc.inverse_transform(np.array(lstm_scaled).reshape(-1,1)).ravel()

    # Random-Forest
    rf = RandomForestRegressor(rf_estimators, random_state=42, min_samples_leaf=2)
    rf_train = make_lags(df[col], rf_lags)
    rf.fit(rf_train.drop(columns="y"), rf_train["y"])
    vals = df[col].iloc[-rf_lags:].tolist(); rf_pred=[]
    for _ in range(forecast_years):
        p = rf.predict(np.array(vals[-rf_lags:]).reshape(1,-1))[0]
        rf_pred.append(p); vals.append(p)

    # future index – pass a *string* so year isn’t treated as nanoseconds
    start = int(df.index.year.max()) + 1
    future = pd.date_range(start=f"{start}", periods=forecast_years, freq="YS")

    # Apply formatting rules
    if any(term in title for term in ["Enrollment", "Employee", "Graduate"]):
        lstm_pred = np.round(lstm_pred).astype(int)
        rf_pred = np.round(rf_pred).astype(int)
    else:
        lstm_pred = np.round(lstm_pred, 2)
        rf_pred = np.round(rf_pred, 2)


    lstm_df = pd.DataFrame(lstm_pred, index=future, columns=["LSTM"])
    rf_df   = pd.DataFrame(rf_pred,   index=future, columns=["RF"])

    plt.figure(figsize=(10,4))
    plt.plot(df.index, df[col], "o-", label="Actual")
    plt.plot(lstm_df, "--o", label="LSTM")
    plt.plot(rf_df,   "--x", label="RF")
    plt.title(f"{title}: {future.year[0]}–{future.year[-1]}")
    plt.grid(); plt.legend(); plt.show()
    display(pd.concat([lstm_df, rf_df], axis=1).round(2))

# load chosen file 
csv = PATHS[kpi_to_run]
time_col = "Year" if kpi_to_run in ["R&D Expenditure", "Total Graduates"] else "Date"
df = load_ts(csv, time_col).rename(columns=RENAME.get(kpi_to_run, {}))

# run the right branch 
if kpi_to_run == "Enrollment":
    df["Total"] = df.sum(axis=1)
    forecast(df, "Total", f"{university.upper()} – Total Enrollment")

elif kpi_to_run == "Enrollment Breakdown":
    for c,lbl in ENROLL_MAP.items():
        if c in df.columns: forecast(df[[c]], c, f"{university.upper()} – {lbl}")

elif kpi_to_run == "Employees":
    df["Total"] = df.sum(axis=1)
    forecast(df, "Total", f"{university.upper()} – Total Employees")

elif kpi_to_run == "Employee Breakdown":
    for c,lbl in EMP_MAP.items():
        if c in df.columns: forecast(df[[c]], c, f"{university.upper()} – {lbl}")

elif kpi_to_run == "R&D Expenditure":
    forecast(df, "R&D_Expenditure", f"{university.upper()} – R&D Expenditure")

elif kpi_to_run == "Total Graduates":
    grad_col = next(col for col in df.columns if "Grad" in col or "Total" in col)
    forecast(df, grad_col, f"{university.upper()} – Total Graduates")
