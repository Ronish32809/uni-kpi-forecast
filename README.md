Universities KPI Forecast 

What this is
- A small Streamlit app I made to forecast university KPIs.
- KPIs: Enrollment (+ breakdown), Employees (+ breakdown), R&D Expenditure, Total Graduates.
- Universities covered: Lamar, EOU, UAB.
- It shows the history and then predicts the next N years with two models (LSTM + Random Forest).

How it works
- The app reads CSVs from the data/ folders (data/lamar, data/eastern_oregon, data/uab).
- Each CSV must have either a “Date” or “Year” column.
- I standardize a couple of header names in code (e.g., R&D Expenditure).
- For forecasting I use a simple LSTM (with MinMax scaling + early stopping) and a Random Forest with lag features.
- Counts are rounded to whole numbers; money is shown with 2 decimals.

Run locally
1) install deps:  pip install -r requirements.txt
2) start app:     python -m streamlit run app/dashboard.py

Repo layout
- app/dashboard.py     (Streamlit UI)
- app/forecasting.py   (data loading + models)
- data/...             (CSVs per university)
- requirements.txt

That’s it. I use this to quickly compare LSTM vs RF and visualize the forecasts.
