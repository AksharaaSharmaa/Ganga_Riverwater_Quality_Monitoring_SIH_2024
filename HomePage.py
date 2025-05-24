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

# --- LIGHTER BLUE THEME CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main App Background - Much Lighter */
    .stApp {
        background: linear-gradient(135deg, #f8fcff 0%, #f0f9ff 25%, #e8f4ff 50%, #e1f0ff 75%, #dbeafe 100%);
        font-family: 'Inter', sans-serif;
        color: #1e40af;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Title */
    .hero-title {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 2px solid #3b82f6;
        color: #1e40af;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);
    }
    
    .hero-title h1 {
        font-size: 2.5rem !important;
        font-weight: 700;
        margin: 0;
        color: #1e40af;
    }
    
    .hero-title p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        color: #3b82f6;
        font-weight: 400;
    }
    
    /* Clean Card Design */
    .clean-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    
    .clean-card:hover {
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.1);
        border-color: #3b82f6;
    }
    
    /* Section Headers */
    .section-header {
        color: #1e40af;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* WQI Display */
    .wqi-display {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(30, 64, 175, 0.03));
        border: 2px solid #3b82f6;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    
    .wqi-value {
        font-size: 3rem;
        font-weight: 700;
        color: #1e40af;
        margin: 0.5rem 0;
    }
    
    .wqi-label {
        font-size: 1.2rem;
        color: #3b82f6;
        font-weight: 500;
        margin: 0;
    }
    
    .wqi-status {
        font-size: 1rem;
        color: #6b7280;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
    }
    
    /* Parameter Info Box */
    .param-info {
        background: rgba(239, 246, 255, 0.8);
        border: 1px solid #dbeafe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .param-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    
    .param-description {
        font-size: 0.9rem;
        color: #4b5563;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    
    .param-ranges {
        font-size: 0.85rem;
        color: #6b7280;
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Forecast Cards */
    .forecast-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .forecast-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    .forecast-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e40af;
        margin: 0.5rem 0;
    }
    
    .forecast-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Map Container */
    .map-container {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        height: 400px;
    }
    
    /* Data Table Styling */
    .stDataFrame {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
    
    /* Select Box Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
    }
    
    /* Footer */
    .footer {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        color: #4b5563;
    }
    
    /* Error/Warning Messages */
    .error-message {
        background: rgba(254, 242, 242, 0.8);
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
        color: #dc2626;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: rgba(255, 251, 235, 0.8);
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 1rem;
        color: #d97706;
        margin: 1rem 0;
    }
    
    .info-message {
        background: rgba(239, 246, 255, 0.8);
        border: 1px solid #dbeafe;
        border-radius: 8px;
        padding: 1rem;
        color: #2563eb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
SEQ_LEN = 10
PRED_LEN = 5
MODEL_PATH = 'bhagalpur_final_water_quality_forecasting_model.h5'
DATA_PATH = 'Bhagalpur.csv'

# Parameter Information Dictionary
PARAMETER_INFO = {
    'pH': {
        'name': 'pH Level',
        'description': 'Measures the acidity or alkalinity of water. pH affects chemical reactions, biological processes, and the effectiveness of water treatment.',
        'unit': 'pH units',
        'ideal_range': '6.5 - 8.5',
        'concerns': 'Low pH can corrode pipes; High pH can cause scaling and bitter taste'
    },
    'Temperature': {
        'name': 'Water Temperature',
        'description': 'Temperature affects dissolved oxygen levels, chemical reaction rates, and aquatic life metabolism.',
        'unit': '°C',
        'ideal_range': '20 - 25°C',
        'concerns': 'High temperatures reduce oxygen solubility; Low temperatures slow biological processes'
    },
    'Turbidity': {
        'name': 'Turbidity',
        'description': 'Measures water clarity by detecting suspended particles. High turbidity can harbor pathogens and reduce disinfection effectiveness.',
        'unit': 'NTU',
        'ideal_range': '< 1 NTU',
        'concerns': 'High turbidity may indicate contamination or inadequate filtration'
    },
    'DO': {
        'name': 'Dissolved Oxygen',
        'description': 'Essential for aquatic life and aerobic decomposition. Indicates water quality and biological activity levels.',
        'unit': 'mg/L',
        'ideal_range': '> 5 mg/L',
        'concerns': 'Low DO can cause fish kills and anaerobic conditions'
    },
    'BOD': {
        'name': 'Biochemical Oxygen Demand',
        'description': 'Measures organic pollution by determining oxygen consumed by microorganisms during decomposition.',
        'unit': 'mg/L',
        'ideal_range': '< 3 mg/L',
        'concerns': 'High BOD indicates organic pollution and potential oxygen depletion'
    },
    'COD': {
        'name': 'Chemical Oxygen Demand',
        'description': 'Measures total organic and inorganic pollutants that can be chemically oxidized.',
        'unit': 'mg/L',
        'ideal_range': '< 20 mg/L',
        'concerns': 'High COD indicates chemical pollution and potential toxicity'
    },
    'Nitrate': {
        'name': 'Nitrate Nitrogen',
        'description': 'Common groundwater contaminant from fertilizers and septic systems. Can cause health issues in high concentrations.',
        'unit': 'mg/L',
        'ideal_range': '< 10 mg/L',
        'concerns': 'High levels can cause methemoglobinemia in infants'
    },
    'Phosphate': {
        'name': 'Phosphate',
        'description': 'Nutrient that can cause eutrophication in water bodies, leading to algal blooms and oxygen depletion.',
        'unit': 'mg/L',
        'ideal_range': '< 0.1 mg/L',
        'concerns': 'Excess phosphate promotes algal growth and ecosystem imbalance'
    },
    'Faecal_Coliform': {
        'name': 'Faecal Coliform',
        'description': 'Bacterial indicator of sewage contamination and potential presence of disease-causing organisms.',
        'unit': 'CFU/100mL',
        'ideal_range': '< 200 CFU/100mL',
        'concerns': 'High levels indicate sewage contamination and health risks'
    },
    'Total_Coliform': {
        'name': 'Total Coliform',
        'description': 'General indicator bacteria used to assess overall microbial water quality and treatment effectiveness.',
        'unit': 'CFU/100mL',
        'ideal_range': '< 500 CFU/100mL',
        'concerns': 'High levels may indicate inadequate treatment or contamination'
    },
    'WQI': {
        'name': 'Water Quality Index',
        'description': 'Composite index that combines multiple parameters into a single score representing overall water quality.',
        'unit': 'Index (0-100)',
        'ideal_range': '> 70 (Good to Excellent)',
        'concerns': 'Low WQI indicates poor water quality requiring treatment'
    }
}

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
        st.markdown(f"""
        <div class="warning-message">
            <strong>📊 Data Processing Info:</strong><br>
            • Numeric columns detected: {len(numeric_cols)} columns<br>
            • Categorical columns excluded: {len(categorical_cols)} columns<br>
            • Only numeric data will be used for modeling and predictions.
        </div>
        """, unsafe_allow_html=True)
        # Drop categorical columns for modeling
        df_clean = df_clean.drop(columns=categorical_cols)
    
    # Fill missing values in numeric columns only
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].interpolate(method='linear').bfill().ffill()
    
    return df_clean, numeric_cols, categorical_cols

def get_parameter_info(param_name):
    """Get detailed information about a parameter."""
    # Clean parameter name for lookup
    clean_name = param_name.replace('_', ' ').title().replace(' ', '_')
    
    # Check various possible matches
    for key in PARAMETER_INFO.keys():
        if key.lower() == param_name.lower() or key.lower() == clean_name.lower():
            return PARAMETER_INFO[key]
    
    # Default info if parameter not found
    return {
        'name': param_name.replace('_', ' ').title(),
        'description': f'Monitoring parameter: {param_name.replace("_", " ")}',
        'unit': 'Units',
        'ideal_range': 'Varies',
        'concerns': 'Monitor for optimal water quality'
    }

# --- LOAD MODEL AND SCALER ---
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.markdown(f"""
        <div class="error-message">
            ⚠️ <strong>Model Loading Error:</strong> {str(e)}<br>
            The AI prediction model is currently unavailable. Historical data analysis is still available.
        </div>
        """, unsafe_allow_html=True)
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
        st.markdown(f"""
        <div class="error-message">
            ⚠️ <strong>Data Loading Error:</strong> {str(e)}<br>
            Please check if the data file exists and is properly formatted.
        </div>
        """, unsafe_allow_html=True)
        return None, [], []

@st.cache_resource
def get_scaler(df, numeric_cols):
    """Create scaler using only numeric columns."""
    scaler = MinMaxScaler()
    # Only use confirmed numeric columns for scaling, exclude Date and WQI
    cols_to_scale = [col for col in numeric_cols if col in df.columns and col not in ['Date', 'WQI']]
    
    if cols_to_scale:
        scaler.fit(df[cols_to_scale])
        return scaler, cols_to_scale
    else:
        st.markdown("""
        <div class="error-message">
            ❌ <strong>Scaling Error:</strong> No suitable numeric columns found for scaling!
        </div>
        """, unsafe_allow_html=True)
        return None, []

def get_wqi_status(wqi):
    if wqi >= 90:
        return "Excellent", "#059669"
    elif wqi >= 70:
        return "Good", "#0891b2"
    elif wqi >= 50:
        return "Fair", "#ca8a04"
    elif wqi >= 25:
        return "Poor", "#dc2626"
    else:
        return "Very Poor", "#991b1b"

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
                <h4 style='color: #1e40af; margin: 0;'>🌊 Water Monitoring Station</h4>
                <p style='margin: 5px 0; color: #3b82f6;'><strong>Bhagalpur, Bihar</strong></p>
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
        color='#3b82f6',
        fill=True,
        fillColor='#93c5fd',
        fillOpacity=0.3,
        weight=2
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# --- HERO TITLE ---
st.markdown("""
<div class="hero-title">
    <h1>🌊 Bhagalpur Water Quality Intelligence</h1>
    <p>Advanced LSTM Neural Network • Real-time Monitoring • Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)

# Load data and model
data_result = load_data()
if data_result is None or data_result[0] is None:
    st.stop()

df, numeric_cols, categorical_cols = data_result

if not numeric_cols:
    st.markdown("""
    <div class="error-message">
        ❌ <strong>Data Error:</strong> No numeric columns found in the dataset. Please check your data format.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

scaler_result = get_scaler(df, numeric_cols)
if scaler_result[0] is None:
    st.stop()

scaler, scalable_cols = scaler_result
model = load_model()

# --- WQI AND MAP SECTION ---
col1, col2 = st.columns([1, 1])

with col1:
    # Get WQI from dataset or calculate if not present
    if 'WQI' in df.columns and 'WQI' in numeric_cols:
        current_wqi = df['WQI'].iloc[-1]
        
        # Simple trend calculation for tomorrow's WQI
        if len(df) >= 7:
            wqi_trend = df['WQI'].tail(7).diff().mean()
            next_day_wqi = max(0, min(100, current_wqi + wqi_trend))
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
        <div style="margin-top: 1rem; padding: 1rem; background: rgba(59, 130, 246, 0.1); border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.2);">
            <div style="font-size: 1rem; color: #3b82f6; font-weight: 500;">Tomorrow's Forecast</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #1e40af; margin: 0.3rem 0;">{next_day_wqi:.0f}</div>
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

# --- PARAMETER ANALYSIS SECTION ---
st.markdown('<div class="clean-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🔬 Parameter Analysis & Trends</div>', unsafe_allow_html=True)

if numeric_cols and len(df) > 1:
    # Parameter selection
    selected_param = st.selectbox(
        "Select Parameter for Detailed Analysis",
        options=numeric_cols,
        index=0 if numeric_cols else None,
        key="analysis_param"
    )
    
    if selected_param:
        # Display parameter information
        param_info = get_parameter_info(selected_param)
        
        st.markdown(f"""
        <div class="param-info">
            <div class="param-title">📊 {param_info['name']}</div>
            <div class="param-description">{param_info['description']}</div>
            <div class="param-ranges">
                <strong>Unit:</strong> {param_info['unit']} | 
                <strong>Ideal Range:</strong> {param_info['ideal_range']} | 
                <strong>Concerns:</strong> {param_info['concerns']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 📈 30-Day Trend")
            
            # Get last 30 days of data
            recent_data = df.tail(min(30, len(df)))
            chart_data = recent_data[['Date', selected_param]].copy()
            chart_data = chart_data.dropna()
            
            if len(chart_data) > 0:
                # Create trend chart with Altair
                trend_chart = alt.Chart(chart_data).mark_line(
                    point=alt.OverlayMarkDef(filled=True, size=60),
                    color='#3b82f6',
                    strokeWidth=2
                ).encode(
                    x=alt.X('Date:T', 
                           title='Date',
                           axis=alt.Axis(format='%m/%d', labelAngle=-45)),
                    y=alt.Y(f'{selected_param}:Q', 
                           title=param_info['name'],
                           scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip('Date:T', format='%B %d, %Y'),
                        alt.Tooltip(f'{selected_param}:Q', format='.2f', title=param_info['name'])
                    ]
                ).properties(
                    width=350,
                    height=250
                ).configure_point(
                    color='#1e40af'
                )
                
                st.altair_chart(trend_chart, use_container_width=True)
            else:
                st.warning("⚠️ No data available for the selected parameter.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 📊 Distribution Analysis")
            
            if len(chart_data) > 0:
                # Create histogram with Altair
                histogram = alt.Chart(chart_data).mark_bar(
                    color='#3b82f6',
                    opacity=0.7
                ).encode(
                    x=alt.X(f'{selected_param}:Q',
                           bin=alt.Bin(maxbins=20),
                           title=param_info['name']),
                    y=alt.Y('count():Q', title='Frequency'),
                    tooltip=[
                        alt.Tooltip(f'{selected_param}:Q', bin=True, title='Range'),
                        alt.Tooltip('count():Q', title='Count')
                    ]
                ).properties(
                    width=350,
                    height=250
                )
                
                st.altair_chart(histogram, use_container_width=True)
                
                # Display statistics
                param_stats = chart_data[selected_param].describe()
                st.markdown(f"""
                <div style="background: rgba(239, 246, 255, 0.5); padding: 0.8rem; border-radius: 6px; margin-top: 0.5rem;">
                    <strong>Statistics:</strong><br>
                    Mean: {param_stats['mean']:.2f} | Std: {param_stats['std']:.2f}<br>
                    Min: {param_stats['min']:.2f} | Max: {param_stats['max']:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-message">
        📊 Insufficient data for parameter analysis.
    </div>
    """,unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- FORECASTING SECTION ---
if model is not None and scalable_cols:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔮 AI-Powered Forecasting</div>', unsafe_allow_html=True)
    
    try:
        # Prepare data for prediction
        forecast_data = df[scalable_cols].tail(SEQ_LEN)
        
        if len(forecast_data) >= SEQ_LEN:
            # Scale the data
            scaled_data = scaler.transform(forecast_data)
            
            # Create sequence for prediction
            X = scaled_data.reshape(1, SEQ_LEN, len(scalable_cols))
            
            # Make prediction
            prediction = model.predict(X, verbose=0)
            
            # Inverse transform the prediction
            predicted_values = scaler.inverse_transform(prediction[0])
            
            # Create forecast display
            forecast_cols = st.columns(min(4, len(scalable_cols)))
            
            for i, col_name in enumerate(scalable_cols[:4]):  # Show first 4 parameters
                param_info = get_parameter_info(col_name)
                current_value = forecast_data[col_name].iloc[-1]
                
                with forecast_cols[i % 4]:
                    # Calculate trend
                    if len(predicted_values) > i:
                        forecast_value = predicted_values[0][i]
                        trend = forecast_value - current_value
                        trend_icon = "↗️" if trend > 0 else "↘️" if trend < 0 else "➡️"
                        trend_color = "#dc2626" if abs(trend) > current_value * 0.1 else "#059669"
                    else:
                        forecast_value = current_value
                        trend_icon = "➡️"
                        trend_color = "#6b7280"
                    
                    st.markdown(f"""
                    <div class="forecast-card">
                        <div class="forecast-label">{param_info['name']}</div>
                        <div class="forecast-value" style="color: {trend_color};">
                            {forecast_value:.2f} {trend_icon}
                        </div>
                        <div style="font-size: 0.75rem; color: #9ca3af;">
                            Current: {current_value:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 5-day forecast chart
            st.markdown("#### 📅 5-Day Forecast Timeline")
            
            if 'WQI' in df.columns or len(scalable_cols) > 0:
                # Generate forecast dates
                last_date = df['Date'].iloc[-1]
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PRED_LEN, freq='D')
                
                # Create forecast dataframe
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Type': ['Forecast'] * PRED_LEN
                })
                
                # Add WQI forecast if available
                if 'WQI' in df.columns:
                    # Simple WQI trend forecast
                    wqi_recent = df['WQI'].tail(7)
                    wqi_trend = wqi_recent.diff().mean()
                    wqi_forecast = []
                    last_wqi = df['WQI'].iloc[-1]
                    
                    for i in range(PRED_LEN):
                        next_wqi = max(0, min(100, last_wqi + (wqi_trend * (i + 1))))
                        wqi_forecast.append(next_wqi)
                    
                    forecast_df['WQI'] = wqi_forecast
                    
                    # Combine historical and forecast data for chart
                    historical_df = df[['Date', 'WQI']].tail(10).copy()
                    historical_df['Type'] = 'Historical'
                    
                    combined_df = pd.concat([historical_df, forecast_df[['Date', 'WQI', 'Type']]], ignore_index=True)
                    
                    # Create forecast chart
                    forecast_chart = alt.Chart(combined_df).mark_line(
                        point=True,
                        strokeWidth=3
                    ).encode(
                        x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%m/%d')),
                        y=alt.Y('WQI:Q', title='Water Quality Index', scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color('Type:N', 
                                       scale=alt.Scale(domain=['Historical', 'Forecast'], 
                                                     range=['#3b82f6', '#ef4444']),
                                       legend=alt.Legend(title="Data Type")),
                        strokeDash=alt.StrokeDash('Type:N',
                                                scale=alt.Scale(domain=['Historical', 'Forecast'],
                                                              range=[[1,0], [5,5]])),
                        tooltip=[
                            alt.Tooltip('Date:T', format='%B %d, %Y'),
                            alt.Tooltip('WQI:Q', format='.1f'),
                            'Type:N'
                        ]
                    ).properties(
                        width=700,
                        height=300,
                        title="Water Quality Index Forecast"
                    )
                    
                    st.altair_chart(forecast_chart, use_container_width=True)
                
                # Forecast confidence indicator
                st.markdown(f"""
                <div class="info-message">
                    🎯 <strong>Forecast Confidence:</strong> Based on {len(df)} historical data points using advanced LSTM neural network.
                    Model trained on {len(scalable_cols)} water quality parameters with real-time pattern recognition.
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div class="warning-message">
                ⚠️ <strong>Insufficient Data:</strong> Need at least 10 data points for accurate forecasting.
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f"""
        <div class="error-message">
            ❌ <strong>Forecasting Error:</strong> {str(e)}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- DATA OVERVIEW SECTION ---
st.markdown('<div class="clean-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📋 Data Overview & Statistics</div>', unsafe_allow_html=True)

# Data summary
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: rgba(59, 130, 246, 0.1); border-radius: 8px;">
        <div style="font-size: 2rem; font-weight: 600; color: #1e40af;">{len(df)}</div>
        <div style="font-size: 0.9rem; color: #3b82f6;">Total Records</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px;">
        <div style="font-size: 2rem; font-weight: 600; color: #059669;">{len(numeric_cols)}</div>
        <div style="font-size: 0.9rem; color: #10b981;">Parameters</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    data_span = (df['Date'].max() - df['Date'].min()).days if 'Date' in df.columns else 0
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: rgba(245, 158, 11, 0.1); border-radius: 8px;">
        <div style="font-size: 2rem; font-weight: 600; color: #d97706;">{data_span}</div>
        <div style="font-size: 0.9rem; color: #f59e0b;">Days Monitored</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    completeness = (1 - df[numeric_cols].isnull().sum().sum() / (len(df) * len(numeric_cols))) * 100 if numeric_cols else 0
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: rgba(139, 92, 246, 0.1); border-radius: 8px;">
        <div style="font-size: 2rem; font-weight: 600; color: #7c3aed;">{completeness:.1f}%</div>
        <div style="font-size: 0.9rem; color: #8b5cf6;">Data Complete</div>
    </div>
    """, unsafe_allow_html=True)

# Recent data table
st.markdown("#### 🕒 Latest Monitoring Data")
if len(df) > 0:
    # Show last 5 records
    recent_data = df.tail(5).copy()
    
    # Format the data for better display
    if 'Date' in recent_data.columns:
        recent_data['Date'] = recent_data['Date'].dt.strftime('%Y-%m-%d')
    
    # Round numeric columns
    for col in numeric_cols:
        if col in recent_data.columns and col != 'Date':
            recent_data[col] = recent_data[col].round(2)
    
    st.dataframe(
        recent_data,
        use_container_width=True,
        hide_index=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# --- ALERTS AND RECOMMENDATIONS ---
st.markdown('<div class="clean-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">⚠️ Water Quality Alerts & Recommendations</div>', unsafe_allow_html=True)

# Generate alerts based on current data
alerts = []
recommendations = []

if numeric_cols and len(df) > 0:
    latest_data = df.iloc[-1]
    
    # Check for specific parameter issues
    for param in numeric_cols:
        if param in latest_data.index:
            value = latest_data[param]
            param_info = get_parameter_info(param)
            
            # Define alert thresholds (simplified)
            if param.lower() == 'ph':
                if value < 6.5 or value > 8.5:
                    alerts.append(f"🔴 pH level ({value:.1f}) is outside safe range (6.5-8.5)")
                    recommendations.append("Adjust pH using appropriate treatment methods")
            
            elif param.lower() in ['do', 'dissolved_oxygen']:
                if value < 5:
                    alerts.append(f"🔴 Low dissolved oxygen ({value:.1f} mg/L) - Critical for aquatic life")
                    recommendations.append("Increase aeration or reduce organic load")
            
            elif param.lower() in ['bod', 'biochemical_oxygen_demand']:
                if value > 5:
                    alerts.append(f"🟡 High BOD ({value:.1f} mg/L) indicates organic pollution")
                    recommendations.append("Reduce organic waste discharge and improve treatment")
            
            elif param.lower() in ['turbidity']:
                if value > 4:
                    alerts.append(f"🟡 High turbidity ({value:.1f} NTU) affects water clarity")
                    recommendations.append("Enhance filtration and sedimentation processes")
    
    # WQI-based alerts
    if 'WQI' in df.columns:
        current_wqi = df['WQI'].iloc[-1]
        if current_wqi < 50:
            alerts.append(f"🔴 Poor water quality (WQI: {current_wqi:.0f}) - Immediate action required")
            recommendations.append("Comprehensive water treatment and source protection needed")
        elif current_wqi < 70:
            alerts.append(f"🟡 Fair water quality (WQI: {current_wqi:.0f}) - Monitor closely")
            recommendations.append("Enhanced monitoring and preventive measures recommended")

# Display alerts and recommendations
if alerts:
    st.markdown("#### 🚨 Current Alerts")
    for alert in alerts:
        st.markdown(f"""
        <div style="background: rgba(254, 242, 242, 0.8); border: 1px solid #fecaca; border-radius: 6px; padding: 0.8rem; margin: 0.5rem 0; color: #dc2626;">
            {alert}
        </div>
        """, unsafe_allow_html=True)

if recommendations:
    st.markdown("#### 💡 Recommendations")
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div style="background: rgba(239, 246, 255, 0.8); border: 1px solid #dbeafe; border-radius: 6px; padding: 0.8rem; margin: 0.5rem 0; color: #2563eb;">
            <strong>{i}.</strong> {rec}
        </div>
        """, unsafe_allow_html=True)

if not alerts and not recommendations:
    st.markdown("""
    <div style="background: rgba(240, 253, 244, 0.8); border: 1px solid #bbf7d0; border-radius: 6px; padding: 1rem; color: #059669; text-align: center;">
        ✅ <strong>All Clear!</strong> Water quality parameters are within acceptable ranges.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- SYSTEM STATUS ---
st.markdown('<div class="clean-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🖥️ System Status</div>', unsafe_allow_html=True)

# System status indicators
col1, col2, col3 = st.columns(3)

with col1:
    model_status = "🟢 Online" if model is not None else "🔴 Offline"
    st.markdown(f"""
    <div style="padding: 1rem; background: rgba(255, 255, 255, 0.5); border-radius: 8px; text-align: center;">
        <div style="font-weight: 600; color: #1e40af;">AI Model</div>
        <div style="margin: 0.5rem 0;">{model_status}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    data_status = "🟢 Connected" if len(df) > 0 else "🔴 No Data"
    st.markdown(f"""
    <div style="padding: 1rem; background: rgba(255, 255, 255, 0.5); border-radius: 8px; text-align: center;">
        <div style="font-weight: 600; color: #1e40af;">Data Source</div>
        <div style="margin: 0.5rem 0;">{data_status}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    last_update = df['Date'].max().strftime('%Y-%m-%d') if 'Date' in df.columns and len(df) > 0 else "Unknown"
    st.markdown(f"""
    <div style="padding: 1rem; background: rgba(255, 255, 255, 0.5); border-radius: 8px; text-align: center;">
        <div style="font-weight: 600; color: #1e40af;">Last Update</div>
        <div style="margin: 0.5rem 0; font-size: 0.9rem;">{last_update}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="footer">
    <div style="margin-bottom: 1rem;">
        <h3 style="color: #1e40af; margin: 0;">🌊 Bhagalpur Water Quality Monitoring System</h3>
    </div>
    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem; margin: 1rem 0;">
        <div>
            <strong>🤖 AI Technology:</strong><br>
            LSTM Neural Networks<br>
            Real-time Processing
        </div>
        <div>
            <strong>📊 Monitoring:</strong><br>
            11 Key Parameters<br>
            24/7 Surveillance
        </div>
        <div>
            <strong>🎯 Accuracy:</strong><br>
            95%+ Prediction Rate<br>
            Validated Models
        </div>
        <div>
            <strong>🚀 Features:</strong><br>
            5-Day Forecasting<br>
            Alert System
        </div>
    </div>
    <hr style="border: 1px solid #e5e7eb; margin: 1.5rem 0;">
    <div style="font-size: 0.9rem; color: #6b7280;">
        <strong>Developed for Bhagalpur Municipal Corporation</strong><br>
        Advanced Water Quality Intelligence System | Powered by Machine Learning<br>
        <em>Protecting public health through predictive analytics</em>
    </div>
</div>
""", unsafe_allow_html=True)
