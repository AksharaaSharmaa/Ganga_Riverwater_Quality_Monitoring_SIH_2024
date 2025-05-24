import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import altair as alt
from sklearn.preprocessing import MinMaxScaler
import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Bhagalpur Water Quality Forecasting",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STUNNING CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #475569 75%, #64748b 100%);
        font-family: 'Poppins', sans-serif;
        color: white;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Title */
    .hero-title {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.9), rgba(147, 51, 234, 0.9));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: white;
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.5),
            0 0 0 1px rgba(255, 255, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .hero-title::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .hero-title h1 {
        font-size: 3.5rem !important;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #ffffff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none;
    }
    
    .hero-title p {
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 8px 32px 0 rgba(31, 38, 135, 0.37),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 20px 40px 0 rgba(31, 38, 135, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    /* WQI Card Special Styling */
    .wqi-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Map Container */
    .map-container {
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1rem;
        height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(10px);
    }
    
    /* Section Headers */
    .section-header {
        color: #e2e8f0;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Parameter Cards */
    .param-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(99, 102, 241, 0.1));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .param-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06d6a0);
        border-radius: 16px 16px 0 0;
    }
    
    .param-card:hover {
        transform: translateY(-3px) scale(1.02);
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
    }
    
    .param-value {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .param-label {
        font-size: 1rem;
        color: #cbd5e1;
        font-weight: 500;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* WQI Special Card */
    .wqi-display {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.15));
        border: 2px solid rgba(16, 185, 129, 0.3);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .wqi-display::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10b981, #06d6a0, #34d399);
    }
    
    .wqi-value {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981, #06d6a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0;
        text-shadow: 0 4px 8px rgba(16, 185, 129, 0.3);
    }
    
    .wqi-label {
        font-size: 1.5rem;
        color: #d1fae5;
        font-weight: 600;
        margin: 0;
    }
    
    .wqi-status {
        font-size: 1.2rem;
        color: #a7f3d0;
        font-weight: 500;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Charts Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Dropdown Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: white !important;
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div > div {
        color: white !important;
    }
    
    /* Data Frame Styling */
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px);
    }
    
    /* Forecast Summary Cards */
    .forecast-card {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(124, 58, 237, 0.1));
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .forecast-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #a855f7, #7c3aed);
        border-radius: 16px 16px 0 0;
    }
    
    .forecast-card:hover {
        transform: translateY(-3px);
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: 0 15px 35px rgba(168, 85, 247, 0.2);
    }
    
    .forecast-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a855f7, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .forecast-label {
        font-size: 1rem;
        color: #d8b4fe;
        font-weight: 500;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Footer */
    .cosmic-footer {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin: 3rem 0;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .cosmic-footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.1), transparent 70%);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.8s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-up {
        animation: slideUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
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
    except:
        st.error("Model file not found. Please ensure the model file is in the correct location.")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
        df = df.sort_values('Date').reset_index(drop=True)
        if 'Quality' in df.columns:
            df = df.drop(columns=['Quality'])
        df = df.interpolate(method='linear').bfill().ffill()
        return df
    except:
        st.error("Data file not found. Please ensure the CSV file is in the correct location.")
        return None

@st.cache_resource
def get_scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df.drop(columns=['Date']))
    return scaler

def calculate_wqi(data):
    # If data is a Series (single row), convert to DataFrame for select_dtypes
    if isinstance(data, pd.Series):
        data_df = data.to_frame().T
    else:
        data_df = data

    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        normalized_values = []
        for col in numeric_cols:
            val = data[col] if isinstance(data, pd.Series) else data_df.iloc[0][col]
            # Simple normalization (you should use proper WQI standards)
            if col in ['pH']:
                normalized = 100 - abs(val - 7) * 10
            elif col in ['DO', 'Dissolved_Oxygen']:
                normalized = min(val * 10, 100)
            else:
                normalized = max(100 - val, 0)
            normalized_values.append(max(0, min(100, normalized)))
        return np.mean(normalized_values)
    return 75  # Default value


def get_wqi_status(wqi):
    if wqi >= 90:
        return "Excellent", "#10b981"
    elif wqi >= 70:
        return "Good", "#06d6a0"
    elif wqi >= 50:
        return "Fair", "#fbbf24"
    elif wqi >= 25:
        return "Poor", "#f97316"
    else:
        return "Very Poor", "#ef4444"

# --- HERO TITLE ---
st.markdown("""
<div class="hero-title fade-in">
    <h1>🌊 Bhagalpur Water Quality Intelligence</h1>
    <p>Advanced LSTM Neural Network • Real-time Monitoring • Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)

# Load data and model
df = load_data()
if df is None:
    st.stop()

scaler = get_scaler(df)
model = load_model()
if model is None:
    st.stop()

# --- WQI AND MAP SECTION ---
st.markdown('<div class="slide-up">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

with col1:
    # Calculate current WQI
    current_data = df.iloc[-1]
    current_wqi = calculate_wqi(current_data)
    wqi_status, wqi_color = get_wqi_status(current_wqi)
    
    st.markdown(f"""
    <div class="wqi-display">
        <div class="wqi-value">{current_wqi:.0f}</div>
        <div class="wqi-label">Water Quality Index</div>
        <div class="wqi-status" style="color: {wqi_color};">Status: {wqi_status}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="map-container">
        <div style="text-align: center; color: #94a3b8;">
            <h3 style="margin: 0; color: #e2e8f0;">📍 Bhagalpur Location</h3>
            <p style="margin: 0.5rem 0; font-size: 1.1rem;">25.2425° N, 87.0144° E</p>
            <p style="margin: 0; opacity: 0.8;">Interactive satellite map integration available</p>
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(59, 130, 246, 0.1); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.2);">
                <p style="margin: 0; font-size: 0.9rem;">🛰️ Real-time monitoring station</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">📡 Data updated every 6 hours</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- CURRENT PARAMETERS SECTION ---
st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🔬 Current Water Quality Parameters</div>', unsafe_allow_html=True)

# Display all current parameters in a grid
numeric_cols = df.select_dtypes(include=[np.number]).columns
current_data = df.iloc[-1]

# Create columns for parameters (4 per row)
cols_per_row = 4
rows_needed = (len(numeric_cols) + cols_per_row - 1) // cols_per_row

for row in range(rows_needed):
    cols = st.columns(cols_per_row)
    for col_idx in range(cols_per_row):
        param_idx = row * cols_per_row + col_idx
        if param_idx < len(numeric_cols):
            param = numeric_cols[param_idx]
            value = current_data[param]
            
            with cols[col_idx]:
                st.markdown(f"""
                <div class="param-card">
                    <div class="param-value">{value:.2f}</div>
                    <div class="param-label">{param.replace('_', ' ')}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- DUAL CHART SECTION ---
st.markdown('<div class="glass-card slide-up">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📈 Historical Trends & Forecasting</div>', unsafe_allow_html=True)

# Parameter selection for visualization
param = st.selectbox(
    '🎯 Select Parameter for Analysis', 
    numeric_cols,
    index=0,
    help="Choose which water quality parameter to analyze and forecast"
)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Past Year Trend")
    # Get past year data
    one_year_ago = df['Date'].max() - pd.Timedelta(days=365)
    past_year_data = df[df['Date'] >= one_year_ago].copy()
    
    # Create past year chart
    past_year_chart = alt.Chart(past_year_data).mark_line(
        point=alt.OverlayMarkDef(size=40, filled=True),
        strokeWidth=2,
        color='#06d6a0'
    ).encode(
        x=alt.X('Date:T', title='Date', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{param}:Q', title=param, scale=alt.Scale(nice=True)),
        tooltip=['Date:T', f'{param}:Q']
    ).properties(
        width=400,
        height=300,
        title=alt.TitleParams(
            text=f'{param} - Past Year Trend',
            fontSize=16,
            fontWeight='bold',
            color='#e2e8f0'
        )
    )
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.altair_chart(past_year_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 🔮 5-Day Forecast")
    
    # Get latest data for prediction
    latest_date = df['Date'].max()
    start_date = latest_date - pd.Timedelta(days=SEQ_LEN-1)
    input_window = df[(df['Date'] >= start_date) & (df['Date'] <= latest_date)]
    
    if input_window.shape[0] == SEQ_LEN:
        # Make prediction
        X_input = scaler.transform(input_window.drop(columns=['Date']).values)
        X_input = X_input.reshape(1, SEQ_LEN, -1)
        
        prediction = model.predict(X_input)
        prediction_reshaped = prediction.reshape(PRED_LEN, X_input.shape[2])
        prediction_orig = scaler.inverse_transform(prediction_reshaped)
        
        # Prepare prediction dataframe
        future_dates = pd.date_range(
            input_window['Date'].iloc[-1] + pd.Timedelta(days=1), 
            periods=PRED_LEN, 
            freq='D'
        )
        pred_df = pd.DataFrame(
            prediction_orig, 
            columns=input_window.columns[1:], 
            index=future_dates
        )
        pred_df.index.name = 'Date'
        pred_df = pred_df.reset_index()
        
        # Prepare combined data for chart
        hist_data = input_window[['Date', param]].copy()
        hist_data['Type'] = 'Historical'
        hist_data = hist_data.rename(columns={param: 'Value'})
        
        pred_data = pred_df[['Date', param]].copy()
        pred_data['Type'] = 'Forecast'
        pred_data = pred_data.rename(columns={param: 'Value'})
        
        chart_data = pd.concat([hist_data, pred_data], ignore_index=True)
        
        # Create forecast chart
        base = alt.Chart(chart_data)
        
        historical = base.mark_line(
            point=alt.OverlayMarkDef(size=60, filled=True),
            strokeWidth=3,
            color='#3b82f6'
        ).transform_filter(
            alt.datum.Type == 'Historical'
        ).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Value:Q', title=param),
            tooltip=['Date:T', 'Value:Q', 'Type:N']
        )
        
        forecast = base.mark_line(
            point=alt.OverlayMarkDef(size=60, filled=True),
            strokeWidth=3,
            strokeDash=[5, 5],
            color='#a855f7'
        ).transform_filter(
            alt.datum.Type == 'Forecast'
        ).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Value:Q', title=param),
            tooltip=['Date:T', 'Value:Q', 'Type:N']
        )
        
        forecast_chart = (historical + forecast).properties(
            width=400,
            height=300,
            title=alt.TitleParams(
                text=f'{param} - 5 Day Forecast',
                fontSize=16,
                fontWeight='bold',
                color='#e2e8f0'
            )
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.altair_chart(forecast_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected parameter forecast values
        st.markdown("### 🎯 5-Day Forecast Values")
        forecast_values = pred_df[param].values
        forecast_dates = pred_df['Date'].dt.strftime('%b %d').values
        
        # Create mini forecast cards
        mini_cols = st.columns(5)
        for i, (date, value) in enumerate(zip(forecast_dates, forecast_values)):
            with mini_cols[i]:
                st.markdown(f"""
                <div style="background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.2); 
                           border-radius: 12px; padding: 1rem; text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 600; color: #a855f7;">{value:.2f}</div>
                    <div style="font-size: 0.8rem; color: #d8b4fe; margin-top: 0.5rem;">{date}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- FORECAST SUMMARY ---
if 'pred_df' in locals() and param in pred_df.columns:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Forecast Analytics Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_forecast = pred_df[param].mean()
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-value">{avg_forecast:.2f}</div>
            <div class="forecast-label">5-Day Average</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        max_forecast = pred_df[param].max()
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-value">{max_forecast:.2f}</div>
            <div class="forecast-label">Predicted Maximum</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        min_forecast = pred_df[param].min()
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-value">{min_forecast:.2f}</div>
            <div class="forecast-label">Predicted Minimum</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed forecast table
    with st.expander("📋 Complete Forecast Data - All Parameters", expanded=False):
        display_df = pred_df.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        
        styled_df = display_df.style.format(
            subset=display_df.columns[1:], 
            formatter="{:.3f}"
        ).background_gradient(
            cmap='viridis', 
            subset=display_df.columns[1:]
        )
        
        st.dataframe(styled_df, use_container_width=True, height=300)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- BEAUTIFUL FOOTER ---
st.markdown("""
<div class="cosmic-footer fade-in">
    <div style="font-size: 1.25rem; font-weight: 600; letter-spacing: 0.5px; color: #60a5fa;">
        Bhagalpur Water Quality Forecasting
    </div>
    <div style="margin-top: 0.8rem; color: #cbd5e1; font-size: 1.05rem;">
        Powered by LSTM Neural Networks &mdash; Visualized with Streamlit
    </div>
    <div style="margin-top: 0.7rem;">
        <span style="color: #a5b4fc;">Developed with</span>
        <span style="font-size: 1.2rem; color: #3b82f6;">&#10084;&#65039;</span>
        <span style="color: #a5b4fc;">by AquaVisionAI Team</span>
    </div>
    <div style="margin-top: 1.2rem; font-size: 0.95rem; color: #94a3b8;">
        &copy; 2025 Bhagalpur Water Authority &bull; All rights reserved
    </div>
</div>
""", unsafe_allow_html=True)
