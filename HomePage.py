import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
import google.generativeai as genai
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- ENHANCED CUSTOM CSS FOR PROFESSIONAL DESIGN ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Section Headers */
    .section-header {
        color: #2d3748;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 15px;
        padding: 25px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 10px 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Status Cards */
    .status-good {
        background: linear-gradient(135deg, #48bb78, #38a169);
        border-radius: 12px;
        padding: 15px 20px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
    }
    
    .status-moderate {
        background: linear-gradient(135deg, #ed8936, #dd6b20);
        border-radius: 12px;
        padding: 15px 20px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.3);
    }
    
    .status-bad {
        background: linear-gradient(135deg, #f56565, #e53e3e);
        border-radius: 12px;
        padding: 15px 20px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(245, 101, 101, 0.3);
    }
    
    /* Beautiful Table Styling */
    .beautiful-table {
        background: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    
    .table-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 20px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .table-row {
        padding: 15px 20px;
        border-bottom: 1px solid #e2e8f0;
        transition: background-color 0.2s ease;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .table-row:hover {
        background: #f7fafc;
    }
    
    .table-row:last-child {
        border-bottom: none;
    }
    
    .table-cell {
        flex: 1;
        padding: 0 10px;
    }
    
    .table-cell-header {
        font-weight: 600;
        color: #2d3748;
    }
    
    .table-cell-value {
        font-weight: 500;
        color: #4a5568;
    }
    
    /* Insight Cards */
    .insight-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #667eea;
        margin: 20px 0;
        transition: transform 0.2s ease;
    }
    
    .insight-card:hover {
        transform: translateY(-2px);
    }
    
    .insight-header {
        color: #2d3748;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .ai-response {
        background: #f8fafc;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        line-height: 1.6;
        color: #4a5568;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        color: #4a5568;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #667eea !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        margin: 10px;
        padding: 20px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: white;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        transition: border-color 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8, #6b46a3);
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
SEQ_LEN = 10
PRED_LEN = 5
MODEL_PATH = 'bhagalpur_final_water_quality_forecasting_model.h5'
DATA_PATH = 'Bhagalpur.csv'

# Updated Gemini API Key
GEMINI_API_KEY = "AIzaSyAldo6EIJngpc9TRS58sk3JOCC5ib4E858"

# --- WATER QUALITY PARAMETER THRESHOLDS ---
PARAMETER_THRESHOLDS = {
    'pH': {'good': (6.5, 8.5), 'moderate': (6.0, 9.0), 'unit': 'pH units'},
    'Dissolved Oxygen': {'good': (6, 20), 'moderate': (4, 6), 'unit': 'mg/L'},
    'Biochemical Oxygen Demand': {'good': (0, 3), 'moderate': (3, 6), 'unit': 'mg/L'},
    'Temperature': {'good': (15, 25), 'moderate': (10, 30), 'unit': '°C'},
    'Turbidity': {'good': (0, 5), 'moderate': (5, 25), 'unit': 'NTU'},
    'Nitrate': {'good': (0, 10), 'moderate': (10, 50), 'unit': 'mg/L'},
    'Fecal Coliform': {'good': (0, 100), 'moderate': (100, 1000), 'unit': 'MPN/100ml'},
    'Fecal Streptococci': {'good': (0, 50), 'moderate': (50, 200), 'unit': 'MPN/100ml'},
    'Total Coliform': {'good': (0, 100), 'moderate': (100, 500), 'unit': 'MPN/100ml'},
    'WQI': {'good': (76, 100), 'moderate': (51, 75), 'unit': 'Index'},
    'Conductivity': {'good': (0, 400), 'moderate': (400, 800), 'unit': 'μS/cm'},
    'Rainfall': {'good': (0, 100), 'moderate': (100, 200), 'unit': 'mm'},
    'DO': {'good': (6, 20), 'moderate': (4, 6), 'unit': 'mg/L'},
    'BOD': {'good': (0, 3), 'moderate': (3, 6), 'unit': 'mg/L'},
    'FC': {'good': (0, 100), 'moderate': (100, 1000), 'unit': 'MPN/100ml'},
    'FS': {'good': (0, 50), 'moderate': (50, 200), 'unit': 'MPN/100ml'},
    'TC': {'good': (0, 100), 'moderate': (100, 500), 'unit': 'MPN/100ml'},
    'NO3': {'good': (0, 10), 'moderate': (10, 50), 'unit': 'mg/L'},
    'Temp': {'good': (15, 25), 'moderate': (10, 30), 'unit': '°C'},
    'Cond': {'good': (0, 400), 'moderate': (400, 800), 'unit': 'μS/cm'},
    'Quality': {'good': (3, 4), 'moderate': (2, 3), 'unit': 'Category'}
}

def get_parameter_condition(param_name, value):
    """Determine the condition (Good, Moderate, Bad) for a parameter value"""
    if pd.isna(value) or np.isnan(value):
        return 'Unknown', '#6b7280'
    
    if param_name in PARAMETER_THRESHOLDS:
        thresholds = PARAMETER_THRESHOLDS[param_name]
    else:
        matched_key = None
        param_lower = param_name.lower()
        
        mapping = {
            'dissolved oxygen': 'Dissolved Oxygen',
            'biochemical oxygen demand': 'Biochemical Oxygen Demand',
            'fecal coliform': 'Fecal Coliform',
            'fecal streptococci': 'Fecal Streptococci',
            'total coliform': 'Total Coliform',
            'conductivity': 'Conductivity',
            'temperature': 'Temperature',
            'turbidity': 'Turbidity',
            'nitrate': 'Nitrate',
            'rainfall': 'Rainfall',
            'wqi': 'WQI',
            'quality': 'Quality'
        }
        
        for key, standard_name in mapping.items():
            if key in param_lower or param_lower in key:
                if standard_name in PARAMETER_THRESHOLDS:
                    matched_key = standard_name
                    break
        
        if matched_key:
            thresholds = PARAMETER_THRESHOLDS[matched_key]
        else:
            return 'Requires Assessment', '#6b7280'
    
    good_range = thresholds['good']
    moderate_range = thresholds['moderate']
    
    if param_name in ['Dissolved Oxygen', 'DO', 'WQI', 'Quality']:
        if good_range[0] <= value <= good_range[1]:
            return 'Good', '#48bb78'
        elif moderate_range[0] <= value <= moderate_range[1]:
            return 'Moderate', '#ed8936'
        else:
            return 'Bad', '#f56565'
    else:
        if good_range[0] <= value <= good_range[1]:
            return 'Good', '#48bb78'
        elif moderate_range[0] <= value <= moderate_range[1]:
            return 'Moderate', '#ed8936'
        else:
            return 'Bad', '#f56565'

def create_beautiful_prediction_table(pred_df, future_dates):
    """Create a beautiful HTML table for predictions"""
    html_content = """
    <div class="beautiful-table">
        <div class="table-header">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5rem;">📊</span>
                <span>5-Day Water Quality Forecast</span>
            </div>
        </div>
    """
    
    # Table headers
    html_content += """
        <div class="table-row" style="background: #f8fafc; font-weight: 600;">
            <div class="table-cell table-cell-header">Parameter</div>
            <div class="table-cell table-cell-header">Unit</div>
            <div class="table-cell table-cell-header">Day 1</div>
            <div class="table-cell table-cell-header">Day 2</div>
            <div class="table-cell table-cell-header">Day 3</div>
            <div class="table-cell table-cell-header">Day 4</div>
            <div class="table-cell table-cell-header">Day 5</div>
            <div class="table-cell table-cell-header">Trend</div>
        </div>
    """
    
    # Table rows
    for param in pred_df.columns:
        unit = PARAMETER_THRESHOLDS.get(param, {}).get('unit', '')
        values = pred_df[param].values
        trend = "📈" if values[-1] > values[0] else "📉" if values[-1] < values[0] else "➡️"
        
        html_content += f"""
        <div class="table-row">
            <div class="table-cell table-cell-header">{param}</div>
            <div class="table-cell table-cell-value">{unit}</div>
        """
        
        for value in values:
            condition, color = get_parameter_condition(param, value)
            html_content += f"""
            <div class="table-cell">
                <div style="background: {color}; color: white; padding: 8px; border-radius: 8px; text-align: center; font-weight: 500;">
                    {value:.2f}
                </div>
            </div>
            """
        
        html_content += f"""
            <div class="table-cell table-cell-value" style="text-align: center; font-size: 1.2rem;">{trend}</div>
        </div>
        """
    
    html_content += "</div>"
    return html_content

def create_summary_dashboard(pred_df, future_dates):
    """Create a beautiful summary dashboard"""
    summary_data = []
    
    for param in pred_df.columns:
        param_conditions = []
        param_values = []
        
        for value in pred_df[param].values:
            if not pd.isna(value):
                condition, _ = get_parameter_condition(param, value)
                param_conditions.append(condition)
                param_values.append(float(value))
        
        if param_values:
            good_count = param_conditions.count('Good')
            moderate_count = param_conditions.count('Moderate')
            bad_count = param_conditions.count('Bad')
            
            summary_data.append({
                'Parameter': param,
                'Good Days': good_count,
                'Moderate Days': moderate_count,
                'Bad Days': bad_count,
                'Avg Value': np.mean(param_values),
                'Status': 'Excellent' if good_count >= 4 else 'Good' if good_count >= 3 else 'Needs Attention'
            })
    
    # Create HTML dashboard
    html_content = """
    <div class="beautiful-table">
        <div class="table-header">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5rem;">📋</span>
                <span>Water Quality Summary Dashboard</span>
            </div>
        </div>
    """
    
    # Headers
    html_content += """
        <div class="table-row" style="background: #f8fafc; font-weight: 600;">
            <div class="table-cell table-cell-header">Parameter</div>
            <div class="table-cell table-cell-header">Good Days</div>
            <div class="table-cell table-cell-header">Moderate Days</div>
            <div class="table-cell table-cell-header">Bad Days</div>
            <div class="table-cell table-cell-header">Average</div>
            <div class="table-cell table-cell-header">Overall Status</div>
        </div>
    """
    
    # Data rows
    for data in summary_data:
        status_color = '#48bb78' if data['Status'] == 'Excellent' else '#ed8936' if data['Status'] == 'Good' else '#f56565'
        
        html_content += f"""
        <div class="table-row">
            <div class="table-cell table-cell-header">{data['Parameter']}</div>
            <div class="table-cell">
                <div style="background: #48bb78; color: white; padding: 6px 12px; border-radius: 15px; text-align: center; font-weight: 500; display: inline-block;">
                    {data['Good Days']}
                </div>
            </div>
            <div class="table-cell">
                <div style="background: #ed8936; color: white; padding: 6px 12px; border-radius: 15px; text-align: center; font-weight: 500; display: inline-block;">
                    {data['Moderate Days']}
                </div>
            </div>
            <div class="table-cell">
                <div style="background: #f56565; color: white; padding: 6px 12px; border-radius: 15px; text-align: center; font-weight: 500; display: inline-block;">
                    {data['Bad Days']}
                </div>
            </div>
            <div class="table-cell table-cell-value" style="font-weight: 600;">{data['Avg Value']:.2f}</div>
            <div class="table-cell">
                <div style="background: {status_color}; color: white; padding: 8px 16px; border-radius: 20px; text-align: center; font-weight: 600; display: inline-block;">
                    {data['Status']}
                </div>
            </div>
        </div>
        """
    
    html_content += "</div>"
    return html_content

def create_condition_chart(param_name, predicted_values, dates):
    """Create a condition chart showing Good/Moderate/Bad for predicted values"""
    conditions = []
    colors = []
    
    for value in predicted_values:
        condition, color = get_parameter_condition(param_name, value)
        conditions.append(condition)
        colors.append(color)
    
    fig = go.Figure(data=[
        go.Bar(
            x=[d.strftime('%Y-%m-%d') for d in dates],
            y=predicted_values,
            marker_color=colors,
            text=[f'{cond}<br>{val:.2f}' for cond, val in zip(conditions, predicted_values)],
            textposition='auto',
            hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<br><b>Condition:</b> %{text}<extra></extra>',
            marker=dict(
                line=dict(color='rgba(255,255,255,0.8)', width=2)
            )
        )
    ])
    
    fig.update_layout(
        title={
            'text': f'{param_name} - 5-Day Condition Forecast',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}
        },
        xaxis_title='Date',
        yaxis_title=f'{param_name} ({PARAMETER_THRESHOLDS.get(param_name, {}).get("unit", "")})',
        plot_bgcolor='rgba(255, 255, 255, 0.9)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)',
        font=dict(size=12, family='Inter'),
        height=450,
        margin=dict(l=60, r=60, t=80, b=60),
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        )
    )
    
    return fig

# --- GEMINI AI SETUP ---
@st.cache_resource
def setup_gemini():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Error setting up Gemini AI: {str(e)}")
        return None

def get_water_quality_insights(current_data, predicted_data, parameter_name):
    """Generate AI insights using Gemini API"""
    try:
        model = setup_gemini()
        if model is None:
            return "AI insights unavailable - please check API configuration."
        
        current_values = current_data.iloc[-5:].to_dict('records') if len(current_data) >= 5 else current_data.to_dict('records')
        predicted_values = predicted_data.to_dict('records')
        
        prompt = f"""
        As a water quality expert, analyze the following water quality data for Bhagalpur:

        Current Recent Data (last 5 days):
        {json.dumps(current_values, indent=2, default=str)}

        Predicted Data (next 5 days):
        {json.dumps(predicted_values, indent=2, default=str)}

        Currently viewing parameter: {parameter_name}

        Please provide:
        1. Overall water quality assessment
        2. Trend analysis for the selected parameter ({parameter_name})
        3. Health and safety implications
        4. Recommendations for water treatment or usage
        5. Environmental factors that might be affecting these levels
        6. Comparison with WHO/Indian water quality standards where applicable

        Keep the response comprehensive but concise, focusing on actionable insights.
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate AI insights: {str(e)}"

def get_parameter_specific_insights(parameter_name, current_value, predicted_values):
    """Get parameter-specific insights"""
    try:
        model = setup_gemini()
        if model is None:
            return "Parameter insights unavailable."
        
        prompt = f"""
        Provide specific insights about {parameter_name} in water quality:
        
        Current value: {current_value:.2f}
        Predicted values for next 5 days: {predicted_values.tolist()}
        
        Please explain:
        1. What this parameter measures and its significance
        2. Optimal range for safe drinking water
        3. Health effects of current levels
        4. What the predicted trend suggests
        5. Immediate actions if levels are concerning
        
        Be specific and practical in your recommendations.
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate parameter insights: {str(e)}"

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
        df.columns = df.columns.str.strip()
        df = df.interpolate(method='linear').bfill().ffill()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

@st.cache_resource
def get_scaler(df):
    scaler = MinMaxScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    scaler.fit(df[numeric_columns])
    return scaler, numeric_columns

# --- MAIN APPLICATION ---
def main():
    # Header
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🌊 Bhagalpur Water Quality Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    try:
        # Load data and model
        df = load_data()
        scaler, numeric_columns = get_scaler(df)
        model = load_model()
        
        # Sidebar
        with st.sidebar:
            st.markdown("### 📊 Dashboard Navigation")
            st.markdown("#### Available Parameters")
            for col in numeric_columns:
                st.markdown(f"• {col}")
        
        # WQI Display Section
        st.markdown('<div class="section-header">🎯 Current Water Quality Status</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_good = summary_df['Good Days'].sum()
                st.metric("Total Good Conditions", f"{int(total_good)}")
            with col2:
                total_moderate = summary_df['Moderate Days'].sum()
                st.metric("Total Moderate Conditions", f"{int(total_moderate)}")
            with col3:
                total_bad = summary_df['Bad Days'].sum()
                st.metric("Total Bad Conditions", f"{int(total_bad)}")
            with col4:
                total_assessment = summary_df['Assessment Days'].sum()
                st.metric("Need Assessment", f"{int(total_assessment)}")
        else:
            st.warning("No summary data available")
    
    with tab4:
        st.markdown("### Traditional Trend Analysis")
        
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
        ax.set_title(f'{param} Trend Analysis', 
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

    # --- AI INSIGHTS SECTION ---
    st.subheader('🤖 AI-Powered Water Quality Insights', divider='blue')
    
    # Create two columns for different types of insights
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown('<div class="insight-container">', unsafe_allow_html=True)
            st.markdown("**🔍 Parameter-Specific Analysis**")
            
            with st.spinner("Analyzing parameter data..."):
                current_param_value = input_window[param].iloc[-1]
                predicted_param_values = pred_df[param].values
                param_insights = get_parameter_specific_insights(param, current_param_value, predicted_param_values)
            
            st.markdown('<div class="ai-response">', unsafe_allow_html=True)
            st.markdown(param_insights)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="insight-container">', unsafe_allow_html=True)
            st.markdown("**📊 Overall Water Quality Assessment**")
            
            with st.spinner("Generating comprehensive insights..."):
                comprehensive_insights = get_water_quality_insights(input_window, pred_df, param)
            
            st.markdown('<div class="ai-response">', unsafe_allow_html=True)
            st.markdown(comprehensive_insights)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- DETAILED PREDICTION DATA ---
    st.subheader('📊 Detailed Forecast Data', divider='blue')
    with st.expander("🔍 Expand to View All Predicted Parameters"):
        styled_df = pred_df.style.format(formatter="{:.2f}").background_gradient(
            cmap='Blues', subset=pred_df.columns)
        st.dataframe(styled_df, use_container_width=True)

    # --- QUICK INSIGHTS SUMMARY ---
    st.subheader('⚡ Quick Insights Summary', divider='blue')
    
    # Generate quick summary insights
    with st.spinner("Generating quick summary..."):
        try:
            model = setup_gemini()
            if model:
                quick_prompt = f"""
                Provide 3 bullet points summarizing the key insights about {param} water quality parameter:
                Current value: {input_window[param].iloc[-1]:.2f}
                Trend: {'Increasing' if pred_df[param].iloc[-1] > input_window[param].iloc[-1] else 'Decreasing'}
                
                Make it concise and actionable.
                """
                
                quick_response = model.generate_content(quick_prompt)
                st.markdown(quick_response.text)
            else:
                st.info("Quick insights unavailable - check API configuration")
        except:
            st.info("Quick insights temporarily unavailable")

    # --- FOOTER ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem;">
        Powered by LSTM Neural Network & Google Gemini AI • Data Source: Bhagalpur Water Authority
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check that your model file and data file are in the correct location.")
    st.write("Expected files:")
    st.write("- bhagalpur_final_water_quality_forecasting_model.h5")
    st.write("- Bhagalpur.csv")
    st.write("- Valid Gemini API key")
