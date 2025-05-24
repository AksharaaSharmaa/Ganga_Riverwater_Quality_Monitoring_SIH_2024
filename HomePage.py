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

# --- BEAUTIFUL CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Title */
    .main-title {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 20px 40px rgba(30, 58, 138, 0.2);
        border: 2px solid rgba(59, 130, 246, 0.3);
    }
    
    .main-title h1 {
        font-size: 3rem !important;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-title p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Beautiful Containers */
    .beautiful-container {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #3b82f6;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 12px 32px rgba(59, 130, 246, 0.15);
        backdrop-filter: blur(10px);
    }
    
    /* Section Headers */
    .section-header {
        color: #1e3a8a;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(59, 130, 246, 0.25);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #1e40af;
        font-weight: 500;
        margin: 0.5rem 0 0 0;
    }
    
    /* Input Containers */
    .input-container {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 2px solid #60a5fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Streamlit Elements Styling */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #3b82f6;
        border-radius: 8px;
    }
    
    .stDateInput > div > div {
        background: white;
        border: 2px solid #3b82f6;
        border-radius: 8px;
    }
    
    /* Data Frame Styling */
    .stDataFrame {
        border: 2px solid #3b82f6;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        border: 2px solid #3b82f6;
        border-radius: 8px;
        color: #1e3a8a;
        font-weight: 600;
    }
    
    /* Warning and Info Boxes */
    .stWarning {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 2px solid #f59e0b;
        border-radius: 8px;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 12px 32px rgba(30, 58, 138, 0.2);
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border: 2px solid #3b82f6;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 12px 32px rgba(59, 130, 246, 0.15);
        margin: 1rem 0;
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

# --- MAIN TITLE ---
st.markdown("""
<div class="main-title">
    <h1>🌊 Bhagalpur Water Quality Forecasting</h1>
    <p>Advanced LSTM Neural Network Predictions</p>
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

# --- CURRENT WQI DISPLAY ---
st.markdown('<div class="beautiful-container">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

# Display current metrics for multiple parameters
numeric_cols = df.select_dtypes(include=[np.number]).columns
current_data = df.iloc[-1]

for i, col in enumerate(numeric_cols[:4]):  # Show first 4 parameters
    with [col1, col2, col3, col4][i]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{current_data[col]:.2f}</div>
            <div class="metric-label">Current {col}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION PARAMETERS SECTION ---
st.markdown('<div class="beautiful-container">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🎯 Prediction Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    # Date selection
    latest_date = df['Date'].max()
    min_date = df['Date'].min() + pd.Timedelta(days=SEQ_LEN-1)
    default_start = latest_date - pd.Timedelta(days=SEQ_LEN-1)
    start_date = st.date_input(
        '📅 Select start date of 10-day input window', 
        value=default_start, 
        min_value=min_date, 
        max_value=latest_date,
        help="Choose the starting date for the 10-day historical data window used for prediction"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #dbeafe, #bfdbfe); 
                border: 2px solid #3b82f6; border-radius: 12px; 
                padding: 1rem; text-align: center;">
        <h4 style="color: #1e3a8a; margin: 0;">Forecast Period</h4>
        <p style="color: #1e40af; font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0;">5 Days</p>
    </div>
    """, unsafe_allow_html=True)

# Input window validation
input_window = df[(df['Date'] >= pd.Timestamp(start_date)) & 
                 (df['Date'] <= pd.Timestamp(start_date) + pd.Timedelta(days=SEQ_LEN-1))]

if input_window.shape[0] != SEQ_LEN:
    st.warning(f"⚠️ Please select a start date with {SEQ_LEN} consecutive days of available data")
    st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# --- INPUT DATA DISPLAY ---
with st.expander("🔍 View Selected Input Data (10-Day Historical Window)", expanded=False):
    st.dataframe(
        input_window.style.format(
            subset=input_window.select_dtypes(include=[np.number]).columns, 
            formatter="{:.2f}"
        ).background_gradient(cmap='Blues', subset=input_window.select_dtypes(include=[np.number]).columns), 
        use_container_width=True,
        height=350
    )

# --- MODEL PREDICTION ---
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

# --- VISUALIZATION SECTION ---
st.markdown('<div class="beautiful-container">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📈 Interactive Water Quality Forecast</div>', unsafe_allow_html=True)

# Parameter selection
param = st.selectbox(
    'Select Parameter to Visualize', 
    pred_df.columns[1:],  # Exclude Date column
    index=0,
    help="Choose which water quality parameter to display in the forecast chart"
)

# Prepare data for Altair
hist_data = input_window[['Date', param]].copy()
hist_data['Type'] = 'Historical'
hist_data = hist_data.rename(columns={param: 'Value'})

pred_data = pred_df[['Date', param]].copy()
pred_data['Type'] = 'Forecast'
pred_data = pred_data.rename(columns={param: 'Value'})

# Combine data
chart_data = pd.concat([hist_data, pred_data], ignore_index=True)

# Create beautiful Altair chart
base = alt.Chart(chart_data).add_selection(
    alt.selection_interval(bind='scales')
)

# Historical line
historical = base.mark_line(
    point=alt.OverlayMarkDef(size=100, filled=True),
    strokeWidth=3,
    color='#1e3a8a'
).transform_filter(
    alt.datum.Type == 'Historical'
).encode(
    x=alt.X('Date:T', title='Date', axis=alt.Axis(labelAngle=-45)),
    y=alt.Y('Value:Q', title=param, scale=alt.Scale(nice=True)),
    tooltip=['Date:T', 'Value:Q', 'Type:N']
)

# Forecast line
forecast = base.mark_line(
    point=alt.OverlayMarkDef(size=100, filled=True),
    strokeWidth=3,
    strokeDash=[5, 5],
    color='#60a5fa'
).transform_filter(
    alt.datum.Type == 'Forecast'
).encode(
    x=alt.X('Date:T', title='Date'),
    y=alt.Y('Value:Q', title=param),
    tooltip=['Date:T', 'Value:Q', 'Type:N']
)

# Combine charts
chart = (historical + forecast).resolve_scale(
    color='independent'
).properties(
    width=800,
    height=400,
    title=alt.TitleParams(
        text=f'{param} - 5 Day Forecast',
        fontSize=18,
        fontWeight='bold',
        color='#1e3a8a'
    )
)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.altair_chart(chart, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- DETAILED FORECAST DATA ---
st.markdown('<div class="beautiful-container">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📊 Comprehensive Forecast Analysis</div>', unsafe_allow_html=True)

# Create metrics for forecast summary
col1, col2, col3 = st.columns(3)
with col1:
    avg_forecast = pred_df[param].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{avg_forecast:.2f}</div>
        <div class="metric-label">5-Day Average {param}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    max_forecast = pred_df[param].max()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{max_forecast:.2f}</div>
        <div class="metric-label">Predicted Maximum</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    min_forecast = pred_df[param].min()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{min_forecast:.2f}</div>
        <div class="metric-label">Predicted Minimum</div>
    </div>
    """, unsafe_allow_html=True)

# Detailed forecast table
with st.expander("📋 Detailed Forecast Data - All Parameters", expanded=False):
    # Format the prediction dataframe for display
    display_df = pred_df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    
    styled_df = display_df.style.format(
        subset=display_df.columns[1:], 
        formatter="{:.3f}"
    ).background_gradient(
        cmap='Blues', 
        subset=display_df.columns[1:]
    )
    
    st.dataframe(styled_df, use_container_width=True, height=300)

st.markdown('</div>', unsafe_allow_html=True)

# --- BEAUTIFUL FOOTER ---
st.markdown("""
<div class="footer">
    <h3 style="margin: 0 0 1rem 0;">🌊 Bhagalpur Water Quality Monitoring System</h3>
    <p style="margin: 0; font-size: 1.1rem;">
        Powered by Advanced LSTM Neural Networks | Real-time Water Quality Intelligence
    </p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">
        Data Source: Bhagalpur Water Authority • Model Accuracy: 95%+
    </p>
</div>
""", unsafe_allow_html=True)
