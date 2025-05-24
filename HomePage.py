import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
SEQ_LEN = 10
PRED_LEN = 5
MODEL_PATH = 'bhagalpur_final_water_quality_forecasting_model.h5'
DATA_PATH = 'Bhagalpur.csv'

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Water Quality Forecasting System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 10px 0;
    }
    
    .header-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(79, 172, 254, 0.3);
        text-align: center;
    }
    
    .wqi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        margin: 20px 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #667eea;
        margin: 15px 0;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        border: 2px solid #667eea;
        border-radius: 10px;
    }
    
    .stDateInput > div > div {
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        border: 2px solid #667eea;
        border-radius: 10px;
    }
    
    h1 {
        color: white;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    h3 {
        color: #34495e;
        font-weight: 500;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .quality-indicator {
        padding: 10px 20px;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 5px 0;
    }
    
    .excellent { background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); }
    .good { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .fair { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .poor { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); color: #333; }
    .very-poor { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
</style>
""", unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
def calculate_wqi(df):
    """Calculate Water Quality Index based on multiple parameters"""
    # Simplified WQI calculation - adjust weights based on your specific requirements
    if 'pH' in df.columns:
        ph_score = np.where(df['pH'].between(6.5, 8.5), 100, 
                   np.where(df['pH'].between(6.0, 9.0), 80, 50))
    else:
        ph_score = 75
    
    if 'DO' in df.columns:
        do_score = np.where(df['DO'] > 6, 100,
                   np.where(df['DO'] > 4, 80, 50))
    else:
        do_score = 75
    
    if 'BOD' in df.columns:
        bod_score = np.where(df['BOD'] < 3, 100,
                    np.where(df['BOD'] < 6, 80, 50))
    else:
        bod_score = 75
    
    # Combine scores (you can adjust weights)
    wqi = (ph_score * 0.3 + do_score * 0.4 + bod_score * 0.3)
    return wqi

def get_wqi_category(wqi):
    """Get WQI category and color"""
    if wqi >= 90:
        return "Excellent", "#00b09b"
    elif wqi >= 70:
        return "Good", "#4facfe"
    elif wqi >= 50:
        return "Fair", "#f093fb"
    elif wqi >= 25:
        return "Poor", "#ff9a9e"
    else:
        return "Very Poor", "#667eea"

# --- LOAD MODEL AND SCALER ---
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except:
        st.error("⚠️ Model file not found. Please ensure the model file exists.")
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
        st.error("⚠️ Data file not found. Please ensure the CSV file exists.")
        return None

@st.cache_resource
def get_scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df.drop(columns=['Date']))
    return scaler

# --- MAIN APP ---
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>💧 Water Quality Forecasting System</h1>
        <p style="color: white; font-size: 1.2rem; margin: 0;">
            Advanced LSTM-based 5-Day Water Quality Prediction for Bhagalpur
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    if df is None:
        return
    
    model = load_model()
    if model is None:
        return
    
    scaler = get_scaler(df)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 🎛️ Prediction Settings")
        st.markdown("---")
        
        latest_date = df['Date'].max()
        min_date = df['Date'].min() + pd.Timedelta(days=SEQ_LEN-1)
        default_start = latest_date - pd.Timedelta(days=SEQ_LEN-1)
        
        start_date = st.date_input(
            '📅 Select Start Date (10-day window)', 
            value=default_start, 
            min_value=min_date, 
            max_value=latest_date,
            help="Choose the starting date for the 10-day input window"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Model Type:** LSTM Neural Network  
        **Input Window:** {SEQ_LEN} days  
        **Prediction Horizon:** {PRED_LEN} days  
        **Parameters:** All water quality metrics
        """)
    
    # Prepare input window
    input_window = df[(df['Date'] >= pd.Timestamp(start_date)) & 
                     (df['Date'] <= pd.Timestamp(start_date) + pd.Timedelta(days=SEQ_LEN-1))]
    
    if input_window.shape[0] != SEQ_LEN:
        st.error(f"⚠️ Please select a start date such that {SEQ_LEN} consecutive days are available in the data.")
        return
    
    # Calculate WQI for input data
    input_wqi = calculate_wqi(input_window)
    current_avg_wqi = np.mean(input_wqi)
    
    # Make prediction
    X_input = scaler.transform(input_window.drop(columns=['Date']).values)
    X_input = X_input.reshape(1, SEQ_LEN, -1)
    
    prediction = model.predict(X_input, verbose=0)
    prediction_reshaped = prediction.reshape(PRED_LEN, X_input.shape[2])
    prediction_orig = scaler.inverse_transform(prediction_reshaped)
    
    # Prepare future dates and predictions
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
    pred_df.reset_index(inplace=True)
    pred_df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Calculate predicted WQI
    pred_wqi = calculate_wqi(pred_df)
    future_avg_wqi = np.mean(pred_wqi)
    
    # WQI Dashboard
    st.markdown("## 🌊 Water Quality Index Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_category, current_color = get_wqi_category(current_avg_wqi)
        st.markdown(f"""
        <div class="wqi-card">
            <h3>Current WQI</h3>
            <h1>{current_avg_wqi:.1f}</h1>
            <div class="quality-indicator {current_category.lower().replace(' ', '-')}">
                {current_category}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        future_category, future_color = get_wqi_category(future_avg_wqi)
        st.markdown(f"""
        <div class="wqi-card">
            <h3>Predicted WQI</h3>
            <h1>{future_avg_wqi:.1f}</h1>
            <div class="quality-indicator {future_category.lower().replace(' ', '-')}">
                {future_category}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        wqi_change = future_avg_wqi - current_avg_wqi
        change_icon = "📈" if wqi_change > 0 else "📉" if wqi_change < 0 else "➡️"
        st.markdown(f"""
        <div class="wqi-card">
            <h3>WQI Trend</h3>
            <h1>{change_icon}</h1>
            <div class="quality-indicator {'good' if wqi_change >= 0 else 'poor'}">
                {wqi_change:+.1f} points
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # WQI Trend Chart
    st.markdown("### 📈 WQI Trend Analysis")
    
    # Combine historical and predicted WQI
    all_dates = list(input_window['Date']) + list(pred_df['Date'])
    all_wqi = list(input_wqi) + list(pred_wqi)
    
    fig = go.Figure()
    
    # Historical WQI
    fig.add_trace(go.Scatter(
        x=input_window['Date'],
        y=input_wqi,
        mode='lines+markers',
        name='Historical WQI',
        line=dict(color='#4facfe', width=3),
        marker=dict(size=8, color='#4facfe')
    ))
    
    # Predicted WQI
    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_wqi,
        mode='lines+markers',
        name='Predicted WQI',
        line=dict(color='#667eea', width=3, dash='dash'),
        marker=dict(size=8, color='#667eea')
    ))
    
    fig.update_layout(
        title="Water Quality Index: Historical vs Predicted",
        xaxis_title="Date",
        yaxis_title="WQI Score",
        template="plotly_white",
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Predictions
    st.markdown("## 🔮 5-Day Detailed Forecast")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Parameter selection for detailed view
        param_options = [col for col in pred_df.columns if col != 'Date']
        selected_param = st.selectbox(
            '📊 Select Parameter for Detailed Analysis', 
            param_options,
            help="Choose a water quality parameter to visualize"
        )
        
        # Create detailed parameter chart
        fig_param = go.Figure()
        
        # Historical data
        fig_param.add_trace(go.Scatter(
            x=input_window['Date'],
            y=input_window[selected_param],
            mode='lines+markers',
            name=f'Historical {selected_param}',
            line=dict(color='#4facfe', width=3),
            marker=dict(size=8)
        ))
        
        # Predicted data
        fig_param.add_trace(go.Scatter(
            x=pred_df['Date'],
            y=pred_df[selected_param],
            mode='lines+markers',
            name=f'Predicted {selected_param}',
            line=dict(color='#667eea', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig_param.update_layout(
            title=f"{selected_param}: 10-Day History + 5-Day Forecast",
            xaxis_title="Date",
            yaxis_title=selected_param,
            template="plotly_white",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_param, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Prediction Summary")
        for i, (date, wqi) in enumerate(zip(pred_df['Date'], pred_wqi)):
            category, color = get_wqi_category(wqi)
            st.markdown(f"""
            <div class="prediction-card">
                <strong>Day {i+1}</strong><br>
                {date.strftime('%Y-%m-%d')}<br>
                <span style="color: {color}; font-weight: bold;">
                    WQI: {wqi:.1f} ({category})
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    # Input Data Display
    with st.expander("📋 View Input Data (Last 10 Days)", expanded=False):
        input_display = input_window.copy()
        input_display['WQI'] = input_wqi
        st.dataframe(input_display, use_container_width=True)
    
    # Detailed Predictions Table
    with st.expander("📊 Detailed 5-Day Predictions", expanded=False):
        pred_display = pred_df.copy()
        pred_display['WQI'] = pred_wqi
        pred_display['WQI_Category'] = [get_wqi_category(wqi)[0] for wqi in pred_wqi]
        st.dataframe(pred_display, use_container_width=True)
    
    # Model Performance Info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin: 20px 0;">
        <p><strong>🤖 Model Information:</strong> Advanced LSTM Neural Network trained on historical water quality data</p>
        <p><strong>🔄 Update Frequency:</strong> Real-time predictions based on selected input window</p>
        <p><strong>🎯 Accuracy:</strong> Optimized for multi-parameter water quality forecasting</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
