import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime

# --- CONFIGURATION ---
SEQ_LEN = 10
PRED_LEN = 5
MODEL_PATH = 'bhagalpur_final_water_quality_forecasting_model.h5'
DATA_PATH = 'Bhagalpur.csv'

# --- LOAD MODEL AND SCALER ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.drop(columns=['Quality'])
    df = df.interpolate(method='linear').bfill().ffill()
    return df

@st.cache_resource
def get_scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df.drop(columns=['Date']))
    return scaler

# --- STREAMLIT APP ---
st.title('Water Quality 5-Day Forecasting (LSTM Model)')

df = load_data()
scaler = get_scaler(df)
model = load_model()

# --- SELECT INPUT PERIOD ---
st.subheader('Select the last 10 days for prediction')
latest_date = df['Date'].max()
min_date = df['Date'].min() + pd.Timedelta(days=SEQ_LEN-1)
default_start = latest_date - pd.Timedelta(days=SEQ_LEN-1)
start_date = st.date_input('Start date of 10-day window', value=default_start, min_value=min_date, max_value=latest_date)
input_window = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(start_date) + pd.Timedelta(days=SEQ_LEN-1))]

if input_window.shape[0] != SEQ_LEN:
    st.warning(f"Please select a start date such that 10 consecutive days are available in the data.")
    st.stop()

st.write('Input window:')
st.dataframe(input_window)

# --- PREPARE INPUT FOR MODEL ---
X_input = scaler.transform(input_window.drop(columns=['Date']).values)
X_input = X_input.reshape(1, SEQ_LEN, -1)  # (1, 10, features)

# --- MAKE PREDICTION ---
prediction = model.predict(X_input)
prediction_reshaped = prediction.reshape(PRED_LEN, X_input.shape[2])
prediction_orig = scaler.inverse_transform(prediction_reshaped)

# --- PREPARE DATES FOR FUTURE PREDICTION ---
future_dates = pd.date_range(input_window['Date'].iloc[-1] + pd.Timedelta(days=1), periods=PRED_LEN, freq='D')

# --- VISUALIZE ---
st.subheader('5-Day Ahead Forecast')
pred_df = pd.DataFrame(prediction_orig, columns=input_window.columns[1:], index=future_dates)
st.dataframe(pred_df)

# --- PLOT WQI FORECAST (or any other parameter) ---
param = st.selectbox('Select parameter to plot', pred_df.columns)
fig, ax = plt.subplots(figsize=(10, 5))
# Plot past 10 days
ax.plot(input_window['Date'], input_window[param], marker='o', label='Past 10 days')
# Plot prediction
ax.plot(future_dates, pred_df[param], marker='o', linestyle='--', color='orange', label='Forecast (next 5 days)')
ax.set_xlabel('Date')
ax.set_ylabel(param)
ax.set_title(f'{param}: Past 10 Days and 5-Day Forecast')
ax.legend()
st.pyplot(fig)

st.info('Model: LSTM, Input: Past 10 days, Output: Next 5 days, all parameters predicted jointly.')

# --- OPTIONAL: Show all parameters for all days ---
with st.expander("Show all predicted parameters for next 5 days"):
    st.dataframe(pred_df)

