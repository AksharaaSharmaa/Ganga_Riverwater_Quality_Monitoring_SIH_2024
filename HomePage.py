import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime

# --- CUSTOM CSS FOR BLUE THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    .stApp {
        background: #f0f9ff;
    }
    h1 {
        color: #1e3a8a;
        font-family: 'Inter', sans-serif;
        text-align: center;
        font-size: 2.8rem !important;
    }
    .stMetric {
        background: #e0f2fe !important;
        border-radius: 10px;
        padding: 15px;
    }
    .stDateInput, .stSelectbox {
        background: #fff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-1qg05tj {
        color: #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
SEQ_LEN = 10
PRED_LEN = 5
MODEL_PATH = 'bhagalpur_final_water_quality_forecasting_model.h5'
DATA_PATH = 'Bhagalpur.csv'

# --- LOAD MODEL AND SCALER ---
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Check if 'Quality' column exists before dropping
        if 'Quality' in df.columns:
            df = df.drop(columns=['Quality'])
        
        df = df.interpolate(method='linear').bfill().ffill()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

@st.cache_resource
def get_scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df.drop(columns=['Date']))
    return scaler

# --- STREAMLIT APP LAYOUT ---
st.markdown("<h1>Bhagalpur Water Quality Forecasting</h1>", unsafe_allow_html=True)

try:
    # Load data and model
    df = load_data()
    scaler = get_scaler(df)
    model = load_model()
    
    # --- WQI DISPLAY ---
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("Current WQI")
            # Check if WQI column exists, otherwise use the first numeric column
            wqi_columns = [col for col in df.columns if 'WQI' in col.upper()]
            if wqi_columns:
                current_wqi = df.iloc[-1][wqi_columns[0]]
                st.metric(label="", value=f"{current_wqi:.2f}", 
                         help="Latest Water Quality Index measurement")
            else:
                # Use the first numeric column if WQI is not found
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    current_value = df.iloc[-1][numeric_cols[0]]
                    st.metric(label=numeric_cols[0], value=f"{current_value:.2f}", 
                             help=f"Latest {numeric_cols[0]} measurement")
        with col2:
            st.write("")  # Spacer

    # --- MAIN APP SECTION ---
    with st.container():
        st.subheader('🌊 Water Quality Forecast', divider='blue')
        
        # Automatically use the latest 10 days as input window
        latest_date = df['Date'].max()
        start_date = latest_date - pd.Timedelta(days=SEQ_LEN-1)
        
        # Input window
        input_window = df[(df['Date'] >= start_date) & 
                         (df['Date'] <= latest_date)]
        
        if input_window.shape[0] != SEQ_LEN:
            st.error(f"⚠️ Insufficient data for forecasting. Need at least {SEQ_LEN} consecutive days.")
            st.stop()

    # --- MODEL PREDICTION ---
    X_input = scaler.transform(input_window.drop(columns=['Date']).values)
    X_input = X_input.reshape(1, SEQ_LEN, -1)

    prediction = model.predict(X_input)
    prediction_reshaped = prediction.reshape(PRED_LEN, X_input.shape[2])
    prediction_orig = scaler.inverse_transform(prediction_reshaped)

    # Prepare prediction dataframe
    future_dates = pd.date_range(input_window['Date'].iloc[-1] + pd.Timedelta(days=1), 
                                periods=PRED_LEN, 
                                freq='D')
    pred_df = pd.DataFrame(prediction_orig, 
                          columns=input_window.columns[1:], 
                          index=future_dates)

    # --- VISUALIZATION ---
    st.subheader('📈 5-Day Water Quality Forecast', divider='blue')

    # Parameter selection
    param = st.selectbox('Select Parameter to Visualize', 
                        pred_df.columns,
                        index=0,
                        help="Choose which water quality parameter to display")

    # Create styled plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Manual styling to replace seaborn
    ax.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax.set_facecolor('#f0f9ff')
    fig.patch.set_facecolor('white')
    
    # Plot historical data
    ax.plot(input_window['Date'], input_window[param], 
           marker='o', 
           markersize=8,
           linewidth=2,
           color='#1e3a8a',
           label='Historical Data')

    # Plot predictions
    ax.plot(future_dates, pred_df[param], 
           marker='o', 
           markersize=8,
           linewidth=2,
           linestyle='--',
           color='#60a5fa',
           label='5-Day Forecast')

    # Plot aesthetics
    ax.set_xlabel('Date', fontsize=12, labelpad=10)
    ax.set_ylabel(param, fontsize=12, labelpad=10)
    ax.set_title(f'{param} Forecast Comparison', 
                fontsize=16, 
                pad=20, 
                color='#1e3a8a',
                fontweight='bold')
    ax.legend(frameon=True, 
             facecolor='white', 
             edgecolor='#e5e7eb',
             fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Improve date formatting
    fig.autofmt_xdate()
    
    # Add some padding to the plot
    plt.tight_layout()

    # Display plot
    st.pyplot(fig)

    # --- DETAILED PREDICTION DATA ---
    st.subheader('📊 Detailed Forecast Data', divider='blue')
    with st.expander("🔍 Expand to View All Predicted Parameters"):
        styled_df = pred_df.style.format(formatter="{:.2f}").background_gradient(
            cmap='Blues', subset=pred_df.columns)
        st.dataframe(styled_df, use_container_width=True)

    # --- FOOTER ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem;">
        Powered by LSTM Neural Network • Data Source: Bhagalpur Water Authority
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check that your model file and data file are in the correct location.")
    st.write("Expected files:")
    st.write("- bhagalpur_final_water_quality_forecasting_model.h5")
    st.write("- Bhagalpur.csv")
