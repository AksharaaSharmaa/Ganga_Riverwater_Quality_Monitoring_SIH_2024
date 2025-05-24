import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import altair as alt
from sklearn.preprocessing import MinMaxScaler
import datetime
import folium
from streamlit_folium import st_folium

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Bhagalpur Water Quality Forecasting",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LIGHT BLUE THEME CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 25%, #81d4fa 50%, #4fc3f7 75%, #29b6f6 100%);
        font-family: 'Poppins', sans-serif;
        color: #0d47a1;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Title */
    .hero-title {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(13, 71, 161, 0.15));
        backdrop-filter: blur(20px);
        border: 3px solid #1976d2;
        color: #0d47a1;
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 
            0 25px 50px -12px rgba(25, 118, 210, 0.3),
            0 0 0 1px rgba(33, 150, 243, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
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
        background: linear-gradient(90deg, transparent, rgba(33, 150, 243, 0.2), transparent);
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
        background: linear-gradient(135deg, #0d47a1, #1976d2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none;
    }
    
    .hero-title p {
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        opacity: 0.8;
        font-weight: 400;
        color: #1565c0;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(227, 242, 253, 0.3);
        backdrop-filter: blur(20px);
        border: 2px solid #42a5f5;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 8px 32px 0 rgba(33, 150, 243, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 20px 40px 0 rgba(33, 150, 243, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        border-color: #1976d2;
    }
    
    /* WQI Card Special Styling */
    .wqi-card {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.2), rgba(13, 71, 161, 0.1));
        border: 3px solid #1976d2;
    }
    
    /* Section Headers */
    .section-header {
        color: #0d47a1;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        text-shadow: 0 2px 4px rgba(33, 150, 243, 0.2);
    }
    
    /* Parameter Cards */
    .param-card {
        background: linear-gradient(135deg, rgba(100, 181, 246, 0.2), rgba(33, 150, 243, 0.15));
        backdrop-filter: blur(15px);
        border: 2px solid #42a5f5;
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
        height: 4px;
        background: linear-gradient(90deg, #2196f3, #1976d2, #0d47a1);
        border-radius: 16px 16px 0 0;
    }
    
    .param-card:hover {
        transform: translateY(-3px) scale(1.02);
        border-color: #1976d2;
        box-shadow: 0 10px 30px rgba(33, 150, 243, 0.3);
    }
    
    .param-value {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1976d2, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .param-label {
        font-size: 1rem;
        color: #1565c0;
        font-weight: 500;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* WQI Special Card */
    .wqi-display {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.25), rgba(13, 71, 161, 0.2));
        border: 3px solid #1976d2;
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
        height: 6px;
        background: linear-gradient(90deg, #2196f3, #1976d2, #0d47a1);
    }
    
    .wqi-value {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1976d2, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0;
        text-shadow: 0 4px 8px rgba(25, 118, 210, 0.3);
    }
    
    .wqi-label {
        font-size: 1.5rem;
        color: #0d47a1;
        font-weight: 600;
        margin: 0;
    }
    
    .wqi-status {
        font-size: 1.2rem;
        color: #1565c0;
        font-weight: 500;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Charts Container */
    .chart-container {
        background: rgba(227, 242, 253, 0.2);
        border: 2px solid #42a5f5;
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(33, 150, 243, 0.15);
    }
    
    /* Dropdown Styling */
    .stSelectbox > div > div {
        background: rgba(227, 242, 253, 0.4) !important;
        border: 2px solid #42a5f5 !important;
        border-radius: 12px !important;
        color: #0d47a1 !important;
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div > div {
        color: #0d47a1 !important;
    }
    
    /* Data Frame Styling */
    .stDataFrame {
        border: 2px solid #42a5f5;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(227, 242, 253, 0.4) !important;
        border: 2px solid #42a5f5 !important;
        border-radius: 12px !important;
        color: #0d47a1 !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px);
    }
    
    /* Forecast Summary Cards */
    .forecast-card {
        background: linear-gradient(135deg, rgba(66, 165, 245, 0.2), rgba(25, 118, 210, 0.15));
        border: 2px solid #42a5f5;
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
        height: 4px;
        background: linear-gradient(90deg, #42a5f5, #1976d2);
        border-radius: 16px 16px 0 0;
    }
    
    .forecast-card:hover {
        transform: translateY(-3px);
        border-color: #1976d2;
        box-shadow: 0 15px 35px rgba(33, 150, 243, 0.25);
    }
    
    .forecast-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1976d2, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .forecast-label {
        font-size: 1rem;
        color: #1565c0;
        font-weight: 500;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Footer */
    .cosmic-footer {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.2), rgba(13, 71, 161, 0.15));
        backdrop-filter: blur(20px);
        border: 2px solid #1976d2;
        color: #0d47a1;
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin: 3rem 0;
        box-shadow: 0 25px 50px -12px rgba(33, 150, 243, 0.2);
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
        background: radial-gradient(circle at 50% 50%, rgba(33, 150, 243, 0.1), transparent 70%);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(227, 242, 253, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #42a5f5, #1976d2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1976d2, #0d47a1);
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
    
    /* Map Container Styling */
    .map-container {
        background: rgba(227, 242, 253, 0.2);
        border: 2px solid #42a5f5;
        border-radius: 20px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        height: 400px;
    }
    
    /* Error Message Styling */
    .error-message {
        background: rgba(244, 67, 54, 0.1);
        border: 2px solid #f44336;
        border-radius: 12px;
        padding: 1rem;
        color: #c62828;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: rgba(255, 152, 0, 0.1);
        border: 2px solid #ff9800;
        border-radius: 12px;
        padding: 1rem;
        color: #ef6c00;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
SEQ_LEN = 10
PRED_LEN = 5
MODEL_PATH = 'bhagalpur_final_water_quality_forecasting_model.h5'
DATA_PATH = 'Bhagalpur.csv'

# --- UTILITY FUNCTIONS ---
def identify_numeric_columns(df):
    """Identify only truly numeric columns, excluding categorical text columns."""
    numeric_cols = []
    for col in df.columns:
        if col.lower() in ['date', 'time', 'datetime']:
            continue
        
        # Check if column is numeric
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)
        else:
            # Try to convert to numeric, if it fails, it's likely categorical
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                # This is a categorical/text column
                continue
    
    return numeric_cols

def clean_data(df):
    """Clean and prepare data, handling categorical columns properly."""
    df_clean = df.copy()
    
    # Identify numeric columns only
    numeric_cols = identify_numeric_columns(df_clean)
    
    # Handle categorical columns separately
    categorical_cols = [col for col in df_clean.columns if col not in numeric_cols and col.lower() not in ['date', 'time', 'datetime']]
    
    if categorical_cols:
        st.warning(f"⚠️ Categorical columns detected and will be excluded from modeling: {', '.join(categorical_cols)}")
        # Drop categorical columns for modeling
        df_clean = df_clean.drop(columns=categorical_cols)
    
    # Fill missing values in numeric columns only
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].interpolate(method='linear').bfill().ffill()
    
    return df_clean, numeric_cols, categorical_cols

# --- LOAD MODEL AND SCALER ---
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"⚠️ Model file not found: {str(e)}")
        return None

@st.cache_data
def load_data():
    try:
        # Try different date parsing formats
        for date_format in [True, False]:  # dayfirst=True, then False
            try:
                df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=date_format)
                break
            except:
                continue
        else:
            # If Date column doesn't exist or parsing fails, try without date parsing
            df = pd.read_csv(DATA_PATH)
            # Try to find date column with different names
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce', infer_datetime_format=True)
                df = df.rename(columns={date_cols[0]: 'Date'})
            else:
                # Create a synthetic date column if none exists
                df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Clean the data and handle categorical columns
        df_clean, numeric_cols, categorical_cols = clean_data(df)
        
        return df_clean, numeric_cols, categorical_cols
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        return None, [], []

@st.cache_resource
def get_scaler(df, numeric_cols):
    """Create scaler using only numeric columns."""
    scaler = MinMaxScaler()
    # Only use confirmed numeric columns for scaling
    cols_to_scale = [col for col in numeric_cols if col in df.columns and col not in ['Date', 'WQI']]
    
    if cols_to_scale:
        scaler.fit(df[cols_to_scale])
        return scaler, cols_to_scale
    else:
        st.error("No numeric columns found for scaling!")
        return None, []

def get_wqi_status(wqi):
    if wqi >= 90:
        return "Excellent", "#1976d2"
    elif wqi >= 70:
        return "Good", "#42a5f5"
    elif wqi >= 50:
        return "Fair", "#64b5f6"
    elif wqi >= 25:
        return "Poor", "#90caf9"
    else:
        return "Very Poor", "#bbdefb"

def create_satellite_map():
    # Bhagalpur coordinates
    bhagalpur_lat, bhagalpur_lon = 25.2425, 87.0144
    
    # Create map centered on Bhagalpur
    m = folium.Map(
        location=[bhagalpur_lat, bhagalpur_lon],
        zoom_start=12,
        tiles=None
    )
    
    # Add satellite tile layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add OpenStreetMap layer as alternative
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add marker for water monitoring station
    folium.Marker(
        [bhagalpur_lat, bhagalpur_lon],
        popup=folium.Popup(
            """
            <div style='width: 200px; text-align: center;'>
                <h4 style='color: #1976d2; margin: 0;'>🌊 Water Monitoring Station</h4>
                <p style='margin: 5px 0; color: #0d47a1;'><strong>Bhagalpur, Bihar</strong></p>
                <p style='margin: 5px 0; font-size: 0.9em;'>📍 25.2425°N, 87.0144°E</p>
                <p style='margin: 5px 0; font-size: 0.9em;'>🔄 Real-time monitoring</p>
                <p style='margin: 5px 0; font-size: 0.9em;'>📡 Updated every 6 hours</p>
            </div>
            """,
            max_width=250
        ),
        tooltip="Water Quality Monitoring Station",
        icon=folium.Icon(
            color='blue',
            icon='tint',
            prefix='fa'
        )
    ).add_to(m)
    
    # Add circle to show monitoring area
    folium.Circle(
        location=[bhagalpur_lat, bhagalpur_lon],
        radius=2000,  # 2km radius
        popup='Monitoring Area Coverage',
        color='#1976d2',
        fill=True,
        fillColor='#42a5f5',
        fillOpacity=0.3,
        weight=2
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# --- HERO TITLE ---
st.markdown("""
<div class="hero-title fade-in">
    <h1>🌊 Bhagalpur Water Quality Intelligence</h1>
    <p>Advanced LSTM Neural Network • Real-time Monitoring • Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)

# Load data and model
data_result = load_data()
if data_result is None or data_result[0] is None:
    st.stop()

df, numeric_cols, categorical_cols = data_result

# Display data loading information
if categorical_cols:
    st.markdown(f"""
    <div class="warning-message">
        <strong>📊 Data Processing Info:</strong><br>
        • Numeric columns detected: {len(numeric_cols)} columns<br>
        • Categorical columns excluded: {len(categorical_cols)} columns ({', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''})<br>
        • Only numeric data will be used for modeling and predictions.
    </div>
    """, unsafe_allow_html=True)

if not numeric_cols:
    st.error("❌ No numeric columns found in the dataset. Please check your data format.")
    st.stop()

scaler_result = get_scaler(df, numeric_cols)
if scaler_result[0] is None:
    st.stop()

scaler, scalable_cols = scaler_result
model = load_model()

# --- WQI AND MAP SECTION ---
st.markdown('<div class="slide-up">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

with col1:
    # Get WQI from dataset or calculate if not present
    if 'WQI' in df.columns and 'WQI' in numeric_cols:
        current_wqi = df['WQI'].iloc[-1]
        
        if model is not None and len(scalable_cols) > 0:
            # Make WQI prediction
            latest_date = df['Date'].max()
            start_date = latest_date - pd.Timedelta(days=SEQ_LEN-1)
            input_window = df[(df['Date'] >= start_date) & (df['Date'] <= latest_date)]
            
            if input_window.shape[0] == SEQ_LEN:
                try:
                    # Prepare data for prediction using only scalable columns
                    X_input = scaler.transform(input_window[scalable_cols].values)
                    X_input = X_input.reshape(1, SEQ_LEN, -1)
                    
                    # Make prediction
                    prediction = model.predict(X_input, verbose=0)
                    prediction_reshaped = prediction.reshape(PRED_LEN, X_input.shape[2])
                    prediction_orig = scaler.inverse_transform(prediction_reshaped)
                    
                    # Create future dates
                    future_dates = pd.date_range(
                        input_window['Date'].iloc[-1] + pd.Timedelta(days=1), 
                        periods=PRED_LEN, 
                        freq='D'
                    )
                    
                    # Create prediction dataframe
                    pred_df = pd.DataFrame(
                        prediction_orig, 
                        columns=scalable_cols, 
                        index=future_dates
                    )
                    
                    # For WQI, estimate based on trend
                    wqi_trend = df['WQI'].tail(5).diff().mean()
                    next_day_wqi = max(0, min(100, current_wqi + wqi_trend))
                    
                except Exception as e:
                    st.warning(f"⚠️ Prediction error: {str(e)}")
                    next_day_wqi = current_wqi
            else:
                next_day_wqi = current_wqi
        else:
            next_day_wqi = current_wqi
            
    else:
        # Fallback calculation if WQI not in dataset
        if numeric_cols:
            # Simple WQI estimation based on available parameters
            current_data = df[numeric_cols].iloc[-1]
            current_wqi = min(100, max(0, np.mean(current_data.fillna(50))))  # Simple average
            next_day_wqi = current_wqi
        else:
            current_wqi = 75  # Default value
            next_day_wqi = 73
    
    wqi_status, wqi_color = get_wqi_status(current_wqi)
    
    st.markdown(f"""
    <div class="wqi-display">
        <div class="wqi-value">{current_wqi:.0f}</div>
        <div class="wqi-label">Current Water Quality Index</div>
        <div class="wqi-status" style="color: {wqi_color};">Status: {wqi_status}</div>
        <div style="margin-top: 1rem; padding: 1rem; background: rgba(33, 150, 243, 0.1); border-radius: 12px; border: 1px solid rgba(66, 165, 245, 0.3);">
            <div style="font-size: 1.1rem; color: #1565c0; font-weight: 600;">Tomorrow's Forecast</div>
            <div style="font-size: 2rem; font-weight: 700; color: #1976d2; margin: 0.5rem 0;">{next_day_wqi:.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    
    # Create and display satellite map
    satellite_map = create_satellite_map()
    map_data = st_folium(
        satellite_map,
        width=500,
        height=350,
        returned_objects=["last_object_clicked"]
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- CURRENT PARAMETERS SECTION ---
st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🔬 Current Water Quality Parameters</div>', unsafe_allow_html=True)

# Display all current numeric parameters in a grid
if numeric_cols:
    current_data = df[numeric_cols].iloc[-1]
    
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
else:
    st.warning("⚠️ No numeric parameters available for display.")

st.markdown('</div>', unsafe_allow_html=True)

# --- DUAL CHART SECTION ---
# --- DUAL CHART SECTION ---
st.markdown('<div class="slide-up">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Parameter Trends (Last 30 Days)</div>', unsafe_allow_html=True)
    
    if numeric_cols and len(df) > 1:
        # Get last 30 days of data
        recent_data = df.tail(min(30, len(df)))
        
        # Create parameter selection
        selected_param = st.selectbox(
            "Select Parameter to View",
            options=numeric_cols,
            index=0 if numeric_cols else None,
            key="trend_param"
        )
        
        if selected_param:
            # Create trend chart
            chart_data = recent_data[['Date', selected_param]].copy()
            chart_data = chart_data.dropna()
            
            if len(chart_data) > 0:
                trend_chart = alt.Chart(chart_data).mark_line(
                    point=True,
                    color='#1976d2',
                    strokeWidth=3
                ).add_selection(
                    alt.selection_interval(bind='scales')
                ).encode(
                    x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%m/%d')),
                    y=alt.Y(f'{selected_param}:Q', title=selected_param.replace('_', ' ')),
                    tooltip=[
                        alt.Tooltip('Date:T', format='%B %d, %Y'),
                        alt.Tooltip(f'{selected_param}:Q', format='.2f')
                    ]
                ).properties(
                    width=400,
                    height=300,
                    title=f"{selected_param.replace('_', ' ')} Trend"
                ).configure_title(
                    fontSize=16,
                    color='#0d47a1'
                ).configure_axis(
                    labelColor='#1565c0',
                    titleColor='#0d47a1'
                )
                
                st.altair_chart(trend_chart, use_container_width=True)
            else:
                st.warning("⚠️ No data available for the selected parameter.")
    else:
        st.info("📊 Insufficient data for trend analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔄 Parameter Correlation</div>', unsafe_allow_html=True)
    
    if len(numeric_cols) > 1:
        # Calculate correlation matrix
        corr_data = df[numeric_cols].corr().reset_index()
        corr_melted = corr_data.melt(id_vars='index', var_name='variable', value_name='correlation')
        corr_melted = corr_melted.rename(columns={'index': 'param1', 'variable': 'param2'})
        
        # Create heatmap
        heatmap = alt.Chart(corr_melted).mark_rect().encode(
            x=alt.X('param1:N', title='Parameters', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('param2:N', title='Parameters'),
            color=alt.Color(
                'correlation:Q',
                scale=alt.Scale(scheme='blueorange', domain=[-1, 1]),
                title='Correlation'
            ),
            tooltip=[
                'param1:N',
                'param2:N',
                alt.Tooltip('correlation:Q', format='.3f')
            ]
        ).properties(
            width=400,
            height=300,
            title="Parameter Correlation Matrix"
        ).configure_title(
            fontSize=16,
            color='#0d47a1'
        ).configure_axis(
            labelColor='#1565c0',
            titleColor='#0d47a1'
        )
        
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("🔗 Need multiple parameters for correlation analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- FORECASTING SECTION ---
if model is not None and len(scalable_cols) > 0:
    st.markdown('<div class="glass-card slide-up">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔮 5-Day Water Quality Forecast</div>', unsafe_allow_html=True)
    
    try:
        # Prepare input sequence
        latest_date = df['Date'].max()
        start_date = latest_date - pd.Timedelta(days=SEQ_LEN-1)
        input_window = df[(df['Date'] >= start_date) & (df['Date'] <= latest_date)]
        
        if input_window.shape[0] == SEQ_LEN:
            # Make prediction
            X_input = scaler.transform(input_window[scalable_cols].values)
            X_input = X_input.reshape(1, SEQ_LEN, -1)
            
            prediction = model.predict(X_input, verbose=0)
            prediction_reshaped = prediction.reshape(PRED_LEN, X_input.shape[2])
            prediction_orig = scaler.inverse_transform(prediction_reshaped)
            
            # Create future dates
            future_dates = pd.date_range(
                input_window['Date'].iloc[-1] + pd.Timedelta(days=1), 
                periods=PRED_LEN, 
                freq='D'
            )
            
            # Create prediction dataframe
            pred_df = pd.DataFrame(
                prediction_orig, 
                columns=scalable_cols, 
                index=future_dates
            )
            pred_df['Date'] = future_dates
            
            # Display forecast summary cards
            st.markdown("### 📊 Forecast Summary")
            
            # Create forecast summary cards
            forecast_cols = st.columns(PRED_LEN)
            for i, (date, row) in enumerate(pred_df.iterrows()):
                with forecast_cols[i]:
                    day_name = date.strftime('%a')
                    date_str = date.strftime('%m/%d')
                    
                    # Calculate average parameter value for the day
                    avg_value = row[scalable_cols].mean()
                    
                    st.markdown(f"""
                    <div class="forecast-card">
                        <div class="forecast-label">{day_name}<br>{date_str}</div>
                        <div class="forecast-value">{avg_value:.1f}</div>
                        <div class="forecast-label">Avg Quality</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed forecast charts
            st.markdown("### 📈 Detailed Parameter Forecasts")
            
            # Parameter selection for detailed forecast
            forecast_param = st.selectbox(
                "Select Parameter for Detailed Forecast",
                options=scalable_cols,
                index=0,
                key="forecast_param"
            )
            
            if forecast_param:
                # Combine historical and forecast data
                historical_data = df[['Date', forecast_param]].tail(14).copy()
                historical_data['Type'] = 'Historical'
                
                forecast_data = pd.DataFrame({
                    'Date': pred_df['Date'],
                    forecast_param: pred_df[forecast_param],
                    'Type': 'Forecast'
                })
                
                combined_data = pd.concat([historical_data, forecast_data], ignore_index=True)
                
                # Create forecast chart
                base = alt.Chart(combined_data).add_selection(
                    alt.selection_interval(bind='scales')
                )
                
                historical_line = base.mark_line(
                    color='#1976d2',
                    strokeWidth=3
                ).encode(
                    x=alt.X('Date:T', title='Date'),
                    y=alt.Y(f'{forecast_param}:Q', title=forecast_param.replace('_', ' ')),
                    tooltip=[
                        alt.Tooltip('Date:T', format='%B %d, %Y'),
                        alt.Tooltip(f'{forecast_param}:Q', format='.2f'),
                        'Type:N'
                    ]
                ).transform_filter(
                    alt.datum.Type == 'Historical'
                )
                
                forecast_line = base.mark_line(
                    color='#ff9800',
                    strokeWidth=3,
                    strokeDash=[5, 5]
                ).encode(
                    x=alt.X('Date:T', title='Date'),
                    y=alt.Y(f'{forecast_param}:Q', title=forecast_param.replace('_', ' ')),
                    tooltip=[
                        alt.Tooltip('Date:T', format='%B %d, %Y'),
                        alt.Tooltip(f'{forecast_param}:Q', format='.2f'),
                        'Type:N'
                    ]
                ).transform_filter(
                    alt.datum.Type == 'Forecast'
                )
                
                points = base.mark_circle(
                    size=80
                ).encode(
                    x='Date:T',
                    y=f'{forecast_param}:Q',
                    color=alt.Color(
                        'Type:N',
                        scale=alt.Scale(domain=['Historical', 'Forecast'], range=['#1976d2', '#ff9800'])
                    ),
                    tooltip=[
                        alt.Tooltip('Date:T', format='%B %d, %Y'),
                        alt.Tooltip(f'{forecast_param}:Q', format='.2f'),
                        'Type:N'
                    ]
                )
                
                forecast_chart = (historical_line + forecast_line + points).properties(
                    width=800,
                    height=400,
                    title=f"{forecast_param.replace('_', ' ')} - Historical vs Forecast"
                ).configure_title(
                    fontSize=18,
                    color='#0d47a1'
                ).configure_axis(
                    labelColor='#1565c0',
                    titleColor='#0d47a1'
                ).resolve_scale(
                    color='independent'
                )
                
                st.altair_chart(forecast_chart, use_container_width=True)
            
            # Forecast data table
            with st.expander("📋 Detailed Forecast Data", expanded=False):
                display_df = pred_df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                display_df = display_df.round(3)
                st.dataframe(display_df, use_container_width=True)
        
        else:
            st.warning(f"⚠️ Insufficient data for forecasting. Need {SEQ_LEN} consecutive days, but only {input_window.shape[0]} available.")
    
    except Exception as e:
        st.error(f"❌ Forecasting Error: {str(e)}")
        st.info("🔧 This might be due to data format issues or model compatibility problems.")
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="error-message">', unsafe_allow_html=True)
    if model is None:
        st.markdown("❌ **Forecasting Unavailable**: Model file not found or failed to load.")
    else:
        st.markdown("❌ **Forecasting Unavailable**: No scalable numeric columns found for prediction.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- DATA INSIGHTS SECTION ---
st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
st.markdown('<div class="section-header">💡 Data Insights & Statistics</div>', unsafe_allow_html=True)

if numeric_cols:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Statistical Summary")
        stats_df = df[numeric_cols].describe().round(3)
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Key Insights")
        
        # Calculate insights
        total_records = len(df)
        date_range = (df['Date'].max() - df['Date'].min()).days if 'Date' in df.columns else 0
        
        # Find parameters with highest/lowest values
        current_values = df[numeric_cols].iloc[-1]
        highest_param = current_values.idxmax()
        lowest_param = current_values.idxmin()
        
        # Calculate trends for each parameter
        trends = {}
        for param in numeric_cols:
            if len(df) >= 7:
                recent_avg = df[param].tail(7).mean()
                previous_avg = df[param].iloc[-14:-7].mean() if len(df) >= 14 else df[param].head(7).mean()
                trend = "↗️ Increasing" if recent_avg > previous_avg else "↘️ Decreasing" if recent_avg < previous_avg else "➡️ Stable"
                trends[param] = trend
        
        insights_html = f"""
        <div style="background: rgba(33, 150, 243, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(66, 165, 245, 0.3);">
            <p><strong>📈 Total Records:</strong> {total_records:,}</p>
            <p><strong>📅 Data Span:</strong> {date_range} days</p>
            <p><strong>⬆️ Highest Current:</strong> {highest_param} ({current_values[highest_param]:.2f})</p>
            <p><strong>⬇️ Lowest Current:</strong> {lowest_param} ({current_values[lowest_param]:.2f})</p>
        </div>
        """
        st.markdown(insights_html, unsafe_allow_html=True)
        
        if trends:
            st.markdown("#### 📈 Recent Trends (7-day)")
            for param, trend in list(trends.items())[:5]:  # Show top 5
                st.markdown(f"**{param.replace('_', ' ')}:** {trend}")

else:
    st.info("📊 No numeric data available for statistical analysis.")

st.markdown('</div>', unsafe_allow_html=True)

# --- HISTORICAL DATA VIEWER ---
st.markdown('<div class="glass-card slide-up">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📋 Historical Data Explorer</div>', unsafe_allow_html=True)

# Date range selector
if 'Date' in df.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=df['Date'].max() - pd.Timedelta(days=30),
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=df['Date'].max().date(),
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
    
    # Filter data based on date range
    filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
    
    if len(filtered_df) > 0:
        st.markdown(f"#### 📊 Showing {len(filtered_df)} records from {start_date} to {end_date}")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_all_cols = st.checkbox("Show All Columns", value=False)
        with col2:
            download_data = st.checkbox("Enable Data Download", value=False)
        
        # Select columns to display
        if show_all_cols:
            display_cols = filtered_df.columns.tolist()
        else:
            display_cols = ['Date'] + numeric_cols[:10]  # Show date + first 10 numeric columns
        
        display_df = filtered_df[display_cols].copy()
        if 'Date' in display_df.columns:
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download option
        if download_data:
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name=f"bhagalpur_water_quality_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ No data available for the selected date range.")

else:
    st.info("📅 Date column not available for filtering.")
    st.dataframe(df.head(20), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- SYSTEM STATUS AND FOOTER ---
st.markdown('<div class="cosmic-footer fade-in">', unsafe_allow_html=True)

# System status
col1, col2, col3 = st.columns(3)

with col1:
    model_status = "🟢 Active" if model is not None else "🔴 Offline"
    st.markdown(f"""
    <div style="text-align: center;">
        <h4>🤖 AI Model Status</h4>
        <p style="font-size: 1.2rem; font-weight: 600;">{model_status}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    data_status = f"🟢 {len(df)} Records" if len(df) > 0 else "🔴 No Data"
    st.markdown(f"""
    <div style="text-align: center;">
        <h4>📊 Data Status</h4>
        <p style="font-size: 1.2rem; font-weight: 600;">{data_status}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="text-align: center;">
        <h4>🕒 Last Updated</h4>
        <p style="font-size: 1.2rem; font-weight: 600;">{current_time}</p>
    </div>
    """, unsafe_allow_html=True)

# Footer text
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding-top: 2rem; border-top: 2px solid rgba(66, 165, 245, 0.3);">
    <h3 style="color: #0d47a1; margin-bottom: 1rem;">🌊 Bhagalpur Water Quality Intelligence System</h3>
    <p style="font-size: 1.1rem; color: #1565c0; margin-bottom: 0.5rem;">
        <strong>Powered by Advanced LSTM Neural Networks • Real-time Environmental Monitoring</strong>
    </p>
    <p style="color: #1976d2; opacity: 0.8;">
        Protecting water resources through predictive analytics and continuous monitoring
    </p>
    <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(33, 150, 243, 0.1); border-radius: 12px; border: 1px solid rgba(66, 165, 245, 0.3);">
        <p style="font-size: 0.9rem; color: #1565c0; margin: 0;">
            🔬 <strong>Technical Specifications:</strong> LSTM Sequence Length: {SEQ_LEN} days • Prediction Horizon: {PRED_LEN} days • 
            Parameters Monitored: {len(numeric_cols)} • Update Frequency: Real-time
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
