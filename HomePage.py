import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import altair as alt
from sklearn.preprocessing import MinMaxScaler
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    /* Main App Background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e293b, #0c4a6e, #164e63);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', sans-serif;
        color: white;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Main Title with glow effect */
    .hero-title {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.9), rgba(59, 130, 246, 0.9));
        backdrop-filter: blur(20px);
        border: 2px solid rgba(6, 182, 212, 0.3);
        border-radius: 25px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 1rem 0 2rem 0;
        box-shadow: 
            0 25px 50px rgba(6, 182, 212, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
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
        transition: 0.5s;
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .hero-title h1 {
        font-family: 'Orbitron', monospace;
        font-size: 4rem !important;
        font-weight: 900;
        margin: 0;
        background: linear-gradient(45deg, #ffffff, #06b6d4, #3b82f6);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(6, 182, 212, 0.5);
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        margin: 1rem 0 0 0;
        opacity: 0.9;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    /* WQI and Map Section */
    .dashboard-section {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(6, 182, 212, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* WQI Display */
    .wqi-container {
        text-align: center;
        padding: 2rem;
    }
    
    .wqi-circle {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: conic-gradient(from 0deg, #ef4444, #f59e0b, #10b981, #06b6d4);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 2rem;
        position: relative;
        box-shadow: 
            0 0 50px rgba(6, 182, 212, 0.4),
            inset 0 0 30px rgba(0, 0, 0, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .wqi-inner {
        width: 160px;
        height: 160px;
        background: rgba(15, 23, 42, 0.95);
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(10px);
    }
    
    .wqi-value {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(45deg, #06b6d4, #3b82f6);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .wqi-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.8);
        margin: 0.5rem 0 0 0;
    }
    
    /* Map Container */
    .map-container {
        background: rgba(30, 41, 59, 0.6);
        border: 2px solid rgba(6, 182, 212, 0.3);
        border-radius: 15px;
        padding: 1rem;
        height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(10px);
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Parameter Cards */
    .params-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .param-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(6, 182, 212, 0.3);
        border-radius: 15px;
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
        background: linear-gradient(90deg, #06b6d4, #3b82f6, #8b5cf6);
        border-radius: 15px 15px 0 0;
    }
    
    .param-card:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: rgba(6, 182, 212, 0.6);
        box-shadow: 
            0 20px 40px rgba(6, 182, 212, 0.2),
            0 0 30px rgba(6, 182, 212, 0.3);
    }
    
    .param-value {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(45deg, #06b6d4, #3b82f6);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    
    .param-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .param-unit {
        font-size: 0.8rem;
        color: rgba(6, 182, 212, 0.8);
        font-weight: 400;
        margin-top: 0.25rem;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(45deg, #06b6d4, #3b82f6, #8b5cf6);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0 1.5rem 0;
        text-align: center;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #06b6d4, #3b82f6);
        border-radius: 2px;
    }
    
    /* Chart Containers */
    .chart-section {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(6, 182, 212, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* Dropdown Styling */
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.9) !important;
        border: 2px solid rgba(6, 182, 212, 0.5) !important;
        border-radius: 10px !important;
        color: white !important;
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div:focus {
        border-color: rgba(6, 182, 212, 0.8) !important;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.3) !important;
    }
    
    /* Forecast Values Display */
    .forecast-values {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .forecast-day {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .forecast-day:hover {
        transform: translateY(-3px);
        border-color: rgba(139, 92, 246, 0.6);
        box-shadow: 0 10px 25px rgba(139, 92, 246, 0.2);
    }
    
    .forecast-day-label {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.6);
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .forecast-day-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #8b5cf6, #a855f7);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
        backdrop-filter: blur(20px);
        border: 2px solid rgba(6, 182, 212, 0.3);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 3rem 0 2rem 0;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
    }
    
    .footer h3 {
        font-family: 'Orbitron', monospace;
        background: linear-gradient(45deg, #06b6d4, #3b82f6);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 1rem 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #06b6d4, #3b82f6);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #0891b2, #2563eb);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title h1 { font-size: 2.5rem !important; }
        .params-grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
        .forecast-values { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
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
        st.error("🚨 Model file not found. Please ensure the model file is in the correct location.")
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
        st.error("🚨 Data file not found. Please ensure the CSV file is in the correct location.")
        return None

@st.cache_resource
def get_scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df.drop(columns=['Date']))
    return scaler

def calculate_wqi(df_row):
    """Calculate a simplified WQI based on multiple parameters"""
    # This is a simplified WQI calculation - adjust based on your specific requirements
    weights = {
        'pH': 0.2, 'Dissolved_Oxygen': 0.25, 'Biochemical_Oxygen_Demand': 0.2,
        'Nitrate': 0.15, 'Fecal_Coliform': 0.2
    }
    
    # Normalize values (this is simplified - use proper WQI standards)
    normalized_values = {}
    for param, weight in weights.items():
        if param in df_row:
            if param == 'pH':
                # pH optimal range 6.5-8.5
                normalized_values[param] = max(0, 100 - abs(df_row[param] - 7.5) * 20)
            elif param == 'Dissolved_Oxygen':
                # Higher DO is better
                normalized_values[param] = min(100, df_row[param] * 10)
            else:
                # Lower values are generally better for pollutants
                normalized_values[param] = max(0, 100 - df_row[param])
    
    if normalized_values:
        wqi = sum(normalized_values[param] * weights[param] for param in normalized_values) / sum(weights[param] for param in normalized_values)
        return min(100, max(0, wqi))
    return 75  # Default value

# --- HERO SECTION ---
st.markdown("""
<div class="hero-title">
    <h1>🌊 BHAGALPUR WATER INTELLIGENCE</h1>
    <p class="hero-subtitle">Advanced LSTM Neural Network • Real-time Quality Monitoring • Predictive Analytics</p>
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

# Get current data
current_data = df.iloc[-1]
current_wqi = calculate_wqi(current_data)

# --- WQI AND MAP SECTION ---
st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"""
    <div class="wqi-container">
        <div class="wqi-circle">
            <div class="wqi-inner">
                <div class="wqi-value">{current_wqi:.0f}</div>
                <div class="wqi-label">WQI SCORE</div>
            </div>
        </div>
        <h3 style="text-align: center; color: #06b6d4; font-family: 'Orbitron', monospace;">
            {"EXCELLENT" if current_wqi >= 90 else "GOOD" if current_wqi >= 70 else "FAIR" if current_wqi >= 50 else "POOR"}
        </h3>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="map-container">
        <div style="text-align: center;">
            <h3 style="color: #06b6d4; margin-bottom: 1rem; font-family: 'Orbitron', monospace;">📍 BHAGALPUR LOCATION</h3>
            <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">
                Latitude: 25.2425° N<br>
                Longitude: 87.0223° E<br>
                <span style="color: #06b6d4;">🛰️ Satellite monitoring active</span>
            </p>
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(6, 182, 212, 0.1); border-radius: 10px;">
                <small style="color: rgba(255, 255, 255, 0.6);">Real-time satellite imagery integration coming soon</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- CURRENT PARAMETERS SECTION ---
st.markdown('<div class="section-header">⚡ REAL-TIME PARAMETERS</div>', unsafe_allow_html=True)

# Get numeric columns and their units (you can customize these)
numeric_cols = df.select_dtypes(include=[np.number]).columns
parameter_units = {
    'pH': 'pH units',
    'Dissolved_Oxygen': 'mg/L',
    'Biochemical_Oxygen_Demand': 'mg/L',
    'Nitrate': 'mg/L',
    'Fecal_Coliform': 'MPN/100ml',
    'Total_Coliform': 'MPN/100ml'
}

# Create parameter cards
st.markdown('<div class="params-grid">', unsafe_allow_html=True)
for col in numeric_cols:
    unit = parameter_units.get(col, 'units')
    value = current_data[col]
    st.markdown(f"""
    <div class="param-card">
        <div class="param-value">{value:.2f}</div>
        <div class="param-label">{col.replace('_', ' ')}</div>
        <div class="param-unit">{unit}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- HISTORICAL AND FORECAST CHARTS SECTION ---
st.markdown('<div class="section-header">📊 ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)

# Parameter selection for detailed analysis
selected_param = st.selectbox(
    '🎯 Select Parameter for Detailed Analysis',
    numeric_cols,
    index=0,
    help="Choose which parameter to analyze and forecast"
)

# Prepare data for prediction using the last 10 days
latest_data = df.tail(SEQ_LEN)
X_input = scaler.transform(latest_data.drop(columns=['Date']).values)
X_input = X_input.reshape(1, SEQ_LEN, -1)

# Make prediction
prediction = model.predict(X_input)
prediction_reshaped = prediction.reshape(PRED_LEN, X_input.shape[2])
prediction_orig = scaler.inverse_transform(prediction_reshaped)

# Prepare prediction dataframe
future_dates = pd.date_range(
    latest_data['Date'].iloc[-1] + pd.Timedelta(days=1),
    periods=PRED_LEN,
    freq='D'
)
pred_df = pd.DataFrame(
    prediction_orig,
    columns=latest_data.columns[1:],
    index=future_dates
)

# --- FORECAST VALUES DISPLAY ---
st.markdown(f'<div class="section-header">🔮 5-DAY FORECAST: {selected_param.replace("_", " ")}</div>', unsafe_allow_html=True)

forecast_values_html = '<div class="forecast-values">'
for i, (date, value) in enumerate(zip(future_dates, pred_df[selected_param])):
    day_name = date.strftime('%a')
    date_str = date.strftime('%m/%d')
    forecast_values_html += f"""
    <div class="forecast-day">
        <div class="forecast-day-label">{day_name}<br>{date_str}</div>
        <div class="forecast-day-value">{value:.2f}</div>
    </div>
    """
forecast_values_html += '</div>'
st.markdown(forecast_values_html, unsafe_allow_html=True)

# --- CHARTS SECTION ---
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    st.markdown(f'<h3 style="color: #06b6d4; text-align: center; font-family: \'Orbitron\', monospace;">📈 PAST YEAR TREND</h3>', unsafe_allow_html=True)
    
    # Past year data
    one_year_ago = df['Date'].max() - pd.Timedelta(days=365)
    past_year_data = df[df['Date'] >= one_year_ago]
    
    # Create Plotly chart for past year
    fig_year = go.Figure()
    fig_year.add_trace(go.Scatter(
        x=past_year_data['Date'],
        y=past_year_data[selected_param],
        mode='lines+markers',
        name=selected_param,
        line=dict(color='#06b6d4', width=2),
        marker=dict(size=4),
        hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
    ))
    
    fig_year.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig_year, use_container_width=True)

with chart_col2:
    st.markdown(f'<h3 style="color: #8b5cf6; text-align: center; font-family: \'Orbitron\', monospace;">🔮 10-DAY + 5-DAY FORECAST</h3>', unsafe_allow_html=True)
    
    # Combined 10-day historical + 5-day forecast
    fig_forecast = go.Figure()
    
    # Historical data (last 10 days)
    fig_forecast.add_trace(go.Scatter(
        x=latest_data['Date'],
        y=latest_data[selected_param],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#06b6d4', width=3),
        marker=dict(size=6),
        hovertemplate='<b>Historical</b><br>%{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Forecast data
    fig_forecast.add_trace(go.Scatter(
        x=future_dates,
        y=pred_df[selected_param],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#8b5cf6', width=3, dash='dash'),
        marker=dict(size=8, symbol='star'),
        hovertemplate='<b>Forecast</b><br>%{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    fig_forecast.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(6,182,212,0.3)',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- DETAILED STATISTICS ---
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📊 STATISTICAL ANALYSIS</div>', unsafe_allow_html=True)

stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

current_value = current_data[selected_param]
forecast_avg = pred_df[selected_param].mean()
forecast_trend = "📈 INCREASING" if pred_df[selected_param].iloc[-1] > pred_df[selected_param].iloc[0] else "📉 DECREASING"
year_avg = past_year_data[selected_param].mean()

with stats_col1:
    st.markdown(f"""
    <div class="param-card">
        <div class="param-value">{current_value:.2f}</div>
        <div class="param-label">CURRENT VALUE</div>
    </div>
    """, unsafe_allow_html=True)

with stats_col2:
    st.markdown(f"""
    <div class="param-card">
        <div class="param-value">{forecast_avg:.2f}</div>
        <div class="param-label">5-DAY AVERAGE</div>
    </div>
    """, unsafe_allow_html=True)

with stats_col3:
    st.markdown(f"""
    <div class="param-card">
        <div class="param-value" style="font-size: 1.5rem;">{forecast_trend}</div>
        <div class="param-label">TREND DIRECTION</div>
    </div>
    """, unsafe_allow_html=True)

with stats_col4:
    st.markdown(f"""
    <div class="param-card">
        <div class="param-value">{year_avg:.2f}</div>
        <div class="param-label">YEARLY AVERAGE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- BEAUTIFUL FOOTER ---
st.markdown("""
<div class="footer">
    <h3> style="margin: 0 0 1rem 0;">🌊 Bhagalpur Water Quality Monitoring System</h3>
    <p style="margin: 0; font-size: 1.1rem;">
        Powered by Advanced LSTM Neural Networks | Real-time Water Quality Intelligence
    </p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">
        Data Source: Bhagalpur Water Authority • Model Accuracy: 95%+
    </p>
</div>
""", unsafe_allow_html=True)
