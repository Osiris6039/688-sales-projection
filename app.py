import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from login_config import USERS

# ========== AUTH ==========
def check_login(username, password):
    return username in USERS and USERS[username]["password"] == password

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login Required")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(user, pwd):
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid credentials.")
    st.stop()

# ========== APP ==========
st.title("📊 AI Sales & Customer Forecaster")

DATA_FILE = "stored_data.csv"

# Load or initialize data
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
else:
    df = pd.DataFrame(columns=["Date", "Sales", "Customers", "Weather", "Add_on_Sales"])

# Upload CSV
st.subheader("⬆️ Upload Historical Data")
uploaded_file = st.file_uploader("Choose a CSV", type="csv")
if uploaded_file:
    new_data = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df = pd.concat([df, new_data], ignore_index=True)
    df.drop_duplicates(subset="Date", keep="last", inplace=True)
    df.to_csv(DATA_FILE, index=False)
    st.success("Data uploaded and saved!")

# Manual entry form
st.subheader("📝 Add New Daily Record")
with st.form("data_entry"):
    col1, col2 = st.columns(2)
    date = col1.date_input("Date")
    sales = col2.number_input("Sales", min_value=0)
    customers = col1.number_input("Customers", min_value=0)
    weather = col2.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Stormy"])
    addon = col1.number_input("Add-on Sales", min_value=0)
    submitted = st.form_submit_button("Add Record")
    if submitted:
        df = pd.concat([df, pd.DataFrame([{
            "Date": pd.to_datetime(date),
            "Sales": sales,
            "Customers": customers,
            "Weather": weather,
            "Add_on_Sales": addon
        }])], ignore_index=True)
        df.drop_duplicates(subset="Date", keep="last", inplace=True)
        df.to_csv(DATA_FILE, index=False)
        st.success("New data added and saved!")

# Display current dataset
st.subheader("📄 Current Data")
st.dataframe(df.sort_values("Date"))

# Forecast section
if len(df) >= 14:
    st.subheader("📈 7-Day Forecast")
    df.sort_values("Date", inplace=True)
    df["Weather_Code"] = df["Weather"].astype("category").cat.codes

    features = ["Weather_Code", "Add_on_Sales"]
    df["Day"] = df["Date"].dt.dayofyear
    features += ["Day"]

    X = df[features]
    y_sales = df["Sales"]
    y_cust = df["Customers"]

    X_train, X_test, y_train_s, y_test_s = train_test_split(X, y_sales, test_size=0.2, random_state=42)
    _, _, y_train_c, y_test_c = train_test_split(X, y_cust, test_size=0.2, random_state=42)

    model_sales = XGBRegressor()
    model_sales.fit(X_train, y_train_s)

    model_cust = RandomForestRegressor()
    model_cust.fit(X_train, y_train_c)

    future_dates = pd.date_range(df["Date"].max() + timedelta(days=1), periods=7)
    weather_code = df["Weather_Code"].mode()[0]
    addon_avg = df["Add_on_Sales"].mean()

    future_data = pd.DataFrame({
        "Weather_Code": [weather_code]*7,
        "Add_on_Sales": [addon_avg]*7,
        "Day": future_dates.dayofyear
    })

    forecast_sales = model_sales.predict(future_data)
    forecast_cust = model_cust.predict(future_data)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast_Sales": forecast_sales.round(2),
        "Forecast_Customers": forecast_cust.round(0)
    })

    st.line_chart(forecast_df.set_index("Date"))

    st.dataframe(forecast_df)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
else:
    st.info("Please upload or add at least 14 records to enable forecasting.")