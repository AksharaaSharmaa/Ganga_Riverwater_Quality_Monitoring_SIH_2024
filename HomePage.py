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

# --- CUSTOM CSS FOR BLUE THEME ---
# Beautiful CSS Styling for Water Quality Dashboard
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root Variables for Consistent Theming */
    :root {
        --primary-blue: #0f172a;
        --secondary-blue: #1e293b;
        --accent-blue: #3b82f6;
        --light-blue: #dbeafe;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-success: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --gradient-warning: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --gradient-danger: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
        --glass-bg: rgba(255, 255, 255, 0.15);
        --glass-border: rgba(255, 255, 255, 0.2);
        --shadow-soft: 0 8px 32px rgba(31, 38, 135, 0.37);
        --shadow-strong: 0 20px 40px rgba(0, 0, 0, 0.1);
        --border-radius: 16px;
        --border-radius-small: 8px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Main App Container */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, rgba(120, 219, 255, 0.3) 0%, transparent 50%);
        z-index: -1;
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        33% { transform: translateY(-20px) rotate(1deg); }
        66% { transform: translateY(10px) rotate(-1deg); }
    }
    
    /* Main Content Container */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    h1 {
        font-size: 3.5rem !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        background: linear-gradient(135deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.3)); }
        to { filter: drop-shadow(0 0 30px rgba(255, 255, 255, 0.5)); }
    }
    
    h2 {
        font-size: 2rem !important;
        margin: 2rem 0 1rem 0 !important;
        position: relative;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 0;
        width: 60px;
        height: 4px;
        background: var(--gradient-primary);
        border-radius: 2px;
    }
    
    h3 {
        font-size: 1.5rem !important;
        color: #f8fafc !important;
        margin: 1.5rem 0 1rem 0 !important;
    }
    
    /* Glass Card Styling */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-soft);
        padding: 2rem;
        margin: 1rem 0;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: var(--transition);
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(31, 38, 135, 0.5);
    }
    
    /* Metrics Styling */
    .stMetric {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--border-radius) !important;
        padding: 1.5rem !important;
        text-align: center !important;
        transition: var(--transition) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stMetric:hover {
        transform: translateY(-3px) !important;
        box-shadow: var(--shadow-strong) !important;
    }
    
    .stMetric > div {
        color: #ffffff !important;
    }
    
    .stMetric [data-testid="metric-container"] > div:first-child {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #e2e8f0 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stMetric [data-testid="metric-container"] > div:nth-child(2) {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Input Elements */
    .stSelectbox, .stDateInput, .stNumberInput {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: var(--border-radius) !important;
        border: 1px solid var(--glass-border) !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        transition: var(--transition) !important;
    }
    
    .stSelectbox > div > div {
        background: transparent !important;
        border: none !important;
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div > div {
        background: rgba(15, 23, 42, 0.9) !important;
        color: #ffffff !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--border-radius-small) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: var(--border-radius-small) !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: var(--transition) !important;
        box-shadow: var(--shadow-soft) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-strong) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        padding: 0.5rem;
        margin: 1rem 0;
        border: 1px solid var(--glass-border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #e2e8f0 !important;
        border-radius: var(--border-radius-small) !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
        margin: 0 0.25rem !important;
        transition: var(--transition) !important;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: var(--border-radius) !important;
        border: 1px solid var(--glass-border) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-soft) !important;
    }
    
    .stDataFrame table {
        background: transparent !important;
        color: #ffffff !important;
    }
    
    .stDataFrame th {
        background: rgba(15, 23, 42, 0.7) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 1rem !important;
    }
    
    .stDataFrame td {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 0.75rem 1rem !important;
        color: #e2e8f0 !important;
    }
    
    /* Plotly Charts Container */
    .js-plotly-plot {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: var(--border-radius) !important;
        border: 1px solid var(--glass-border) !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        box-shadow: var(--shadow-soft) !important;
        transition: var(--transition) !important;
    }
    
    .js-plotly-plot:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-strong);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.9) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    .sidebar .sidebar-content {
        background: transparent !important;
        padding: 1rem !important;
    }
    
    /* Condition Status Cards */
    .condition-good {
        background: var(--gradient-success) !important;
        color: #ffffff !important;
        padding: 1rem 1.5rem !important;
        border-radius: var(--border-radius) !important;
        font-weight: 600 !important;
        text-align: center !important;
        margin: 0.5rem !important;
        box-shadow: var(--shadow-soft) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        transition: var(--transition) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .condition-moderate {
        background: var(--gradient-warning) !important;
        color: #ffffff !important;
        padding: 1rem 1.5rem !important;
        border-radius: var(--border-radius) !important;
        font-weight: 600 !important;
        text-align: center !important;
        margin: 0.5rem !important;
        box-shadow: var(--shadow-soft) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        transition: var(--transition) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .condition-bad {
        background: var(--gradient-danger) !important;
        color: #ffffff !important;
        padding: 1rem 1.5rem !important;
        border-radius: var(--border-radius) !important;
        font-weight: 600 !important;
        text-align: center !important;
        margin: 0.5rem !important;
        box-shadow: var(--shadow-soft) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        transition: var(--transition) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .condition-good:hover, .condition-moderate:hover, .condition-bad:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: var(--shadow-strong) !important;
    }
    
    /* Insight Containers */
    .insight-container {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: var(--border-radius) !important;
        padding: 2rem !important;
        box-shadow: var(--shadow-soft) !important;
        border: 1px solid var(--glass-border) !important;
        margin: 1rem 0 !important;
        position: relative !important;
        overflow: hidden !important;
        transition: var(--transition) !important;
    }
    
    .insight-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
    }
    
    .insight-container:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-strong);
    }
    
    .ai-response {
        background: rgba(248, 250, 252, 0.1) !important;
        border-radius: var(--border-radius-small) !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f1f5f9 !important;
        line-height: 1.6 !important;
        font-size: 0.95rem !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: var(--border-radius) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 1px solid var(--glass-border) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid var(--glass-border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--border-radius) var(--border-radius) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
        border-right-color: #3b82f6 !important;
    }
    
    /* Info, Warning, Error, Success */
    .stAlert {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: var(--border-radius) !important;
        border: 1px solid var(--glass-border) !important;
        color: #ffffff !important;
    }
    
    /* Divider */
    hr {
        border: none !important;
        height: 2px !important;
        background: var(--gradient-primary) !important;
        border-radius: 1px !important;
        margin: 2rem 0 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Floating Action Button */
    .fab {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: var(--gradient-primary);
        box-shadow: var(--shadow-strong);
        border: none;
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        transition: var(--transition);
        z-index: 1000;
    }
    
    .fab:hover {
        transform: scale(1.1) rotate(10deg);
    }
    
    /* Loading Animation */
    .loading-shimmer {
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem;
        }
        
        h1 {
            font-size: 2.5rem !important;
        }
        
        .glass-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .stMetric {
            padding: 1rem !important;
        }
        
        .fab {
            bottom: 1rem;
            right: 1rem;
            width: 50px;
            height: 50px;
            font-size: 1.2rem;
        }
    }
    
    /* Scroll Bar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gradient-primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--gradient-secondary);
    }
    
    /* Selection */
    ::selection {
        background: rgba(59, 130, 246, 0.3);
        color: white;
    }
    
    /* Focus States */
    button:focus, input:focus, select:focus {
        outline: 2px solid #3b82f6 !important;
        outline-offset: 2px !important;
    }
    
    /* Print Styles */
    @media print {
        .stApp {
            background: white !important;
        }
        
        .glass-card, .stMetric, .insight-container {
            background: white !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }
    }
    
    /* High Contrast Mode */
    @media (prefers-contrast: high) {
        .glass-card, .stMetric, .insight-container {
            border: 2px solid #ffffff !important;
        }
    }
    
    /* Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
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

# --- UPDATED WATER QUALITY PARAMETER THRESHOLDS ---
# Updated to match your actual parameter names
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
    
    # Alternative naming conventions (in case of slight variations)
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
    # Handle NaN values
    if pd.isna(value) or np.isnan(value):
        return 'Unknown', '#6b7280'
    
    # Try exact match first
    if param_name in PARAMETER_THRESHOLDS:
        thresholds = PARAMETER_THRESHOLDS[param_name]
    else:
        # Try partial matching for common abbreviations
        matched_key = None
        param_lower = param_name.lower()
        
        # Common mappings
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
        
        # Check if parameter name contains any of the mapping keys
        for key, standard_name in mapping.items():
            if key in param_lower or param_lower in key:
                if standard_name in PARAMETER_THRESHOLDS:
                    matched_key = standard_name
                    break
        
        if matched_key:
            thresholds = PARAMETER_THRESHOLDS[matched_key]
        else:
            # Default thresholds for unknown parameters
            # Assume lower is better for most water quality parameters
            return 'Requires Assessment', '#6b7280'
    
    good_range = thresholds['good']
    moderate_range = thresholds['moderate']
    
    # Special handling for parameters where higher is better (like DO, WQI)
    if param_name in ['Dissolved Oxygen', 'DO', 'WQI', 'Quality']:
        # For these parameters, higher values are better
        if good_range[0] <= value <= good_range[1]:
            return 'Good', '#10b981'
        elif moderate_range[0] <= value <= moderate_range[1]:
            return 'Moderate', '#f59e0b'
        else:
            return 'Bad', '#ef4444'
    else:
        # For most parameters, lower values are better
        if good_range[0] <= value <= good_range[1]:
            return 'Good', '#10b981'
        elif moderate_range[0] <= value <= moderate_range[1]:
            return 'Moderate', '#f59e0b'
        else:
            return 'Bad', '#ef4444'

def create_condition_chart(param_name, predicted_values, dates):
    """Create a condition chart showing Good/Moderate/Bad for predicted values"""
    conditions = []
    colors = []
    
    for value in predicted_values:
        condition, color = get_parameter_condition(param_name, value)
        conditions.append(condition)
        colors.append(color)
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=[d.strftime('%Y-%m-%d') for d in dates],
            y=predicted_values,
            marker_color=colors,
            text=[f'{cond}<br>{val:.2f}' for cond, val in zip(conditions, predicted_values)],
            textposition='auto',
            hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<br><b>Condition:</b> %{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f'{param_name} - 5-Day Condition Forecast',
        xaxis_title='Date',
        yaxis_title=f'{param_name} ({PARAMETER_THRESHOLDS.get(param_name, {}).get("unit", "")})',
        plot_bgcolor='rgba(240, 249, 255, 0.8)',
        paper_bgcolor='white',
        font=dict(size=12),
        height=400
    )
    
    return fig

def create_all_parameters_condition_overview(pred_df, future_dates):
    """Create an overview chart showing conditions for all parameters across all days"""
    all_conditions = []
    
    for param in pred_df.columns:
        for i, (date, value) in enumerate(zip(future_dates, pred_df[param].values)):
            condition, _ = get_parameter_condition(param, value)
            # Normalize size values to be positive (add offset and scale)
            normalized_size = abs(float(value)) + 1 if not pd.isna(value) else 1
            all_conditions.append({
                'Parameter': param,
                'Date': date.strftime('%Y-%m-%d'),
                'Day': f'Day {i+1}',
                'Value': float(value) if not pd.isna(value) else 0,
                'Size': normalized_size,
                'Condition': condition
            })
    
    conditions_df = pd.DataFrame(all_conditions)
    
    # Create a heatmap-style visualization
    fig = px.scatter(
        conditions_df, 
        x='Day', 
        y='Parameter',
        color='Condition',
        size='Size',
        color_discrete_map={
            'Good': '#10b981', 
            'Moderate': '#f59e0b', 
            'Bad': '#ef4444',
            'Requires Assessment': '#6b7280'
        },
        title='Water Quality Conditions - All Parameters (5-Day Forecast)',
        hover_data=['Value', 'Date'],
        size_max=20
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(240, 249, 255, 0.8)',
        paper_bgcolor='white',
        height=max(400, len(pred_df.columns) * 40),
        font=dict(size=12)
    )
    
    return fig

def create_condition_summary_table(pred_df, future_dates):
    """Create a summary table showing conditions for all parameters"""
    summary_data = []
    
    for param in pred_df.columns:
        param_conditions = []
        param_values = []
        
        for value in pred_df[param].values:
            if not pd.isna(value):
                condition, _ = get_parameter_condition(param, value)
                param_conditions.append(condition)
                param_values.append(float(value))
        
        if param_values:  # Only process if we have valid values
            # Count conditions
            good_count = param_conditions.count('Good')
            moderate_count = param_conditions.count('Moderate')
            bad_count = param_conditions.count('Bad')
            assessment_count = param_conditions.count('Requires Assessment')
            
            summary_data.append({
                'Parameter': param,
                'Good Days': good_count,
                'Moderate Days': moderate_count,
                'Bad Days': bad_count,
                'Assessment Days': assessment_count,
                'Avg Value': np.mean(param_values),
                'Trend': 'Improving' if param_values[-1] > param_values[0] else 'Declining' if len(param_values) > 1 else 'Stable'
            })
    
    return pd.DataFrame(summary_data)

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
        
        # Prepare data summary for AI analysis
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
        
        # Clean column names (remove extra spaces, standardize naming)
        df.columns = df.columns.str.strip()
        
        # Don't drop Quality column as it might be useful for analysis
        df = df.interpolate(method='linear').bfill().ffill()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

@st.cache_resource
def get_scaler(df):
    scaler = MinMaxScaler()
    # Only fit on numeric columns, excluding Date
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    scaler.fit(df[numeric_columns])
    return scaler, numeric_columns

# --- STREAMLIT APP LAYOUT ---
st.markdown("<h1>🌊 Bhagalpur Water Quality Forecasting</h1>", unsafe_allow_html=True)

try:
    # Load data and model
    df = load_data()
    scaler, numeric_columns = get_scaler(df)
    model = load_model()
    
    # Display available parameters
    st.sidebar.markdown("### Available Parameters")
    for col in numeric_columns:
        st.sidebar.write(f"• {col}")
    
    # --- WQI DISPLAY ---
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.subheader("Current WQI")
            # Check if WQI column exists
            wqi_columns = [col for col in df.columns if 'WQI' in str(col).upper()]
            if wqi_columns:
                current_wqi = df.iloc[-1][wqi_columns[0]]
                wqi_condition, _ = get_parameter_condition('WQI', current_wqi)
                st.metric(label="Water Quality Index", value=f"{current_wqi:.2f}", 
                         help="Latest Water Quality Index measurement")
                st.markdown(f"**Status:** {wqi_condition}")
            else:
                st.info("WQI column not found in dataset")
        
        with col2:
            st.subheader("Data Summary")
            st.metric("Total Parameters", len(numeric_columns))
            st.metric("Latest Date", df['Date'].iloc[-1].strftime('%Y-%m-%d'))
        
        with col3:
            st.subheader("Parameter Coverage")
            # Show which parameters have thresholds defined
            covered_params = []
            uncovered_params = []
            
            for param in numeric_columns:
                if param in PARAMETER_THRESHOLDS or any(key.lower() in param.lower() for key in PARAMETER_THRESHOLDS.keys()):
                    covered_params.append(param)
                else:
                    uncovered_params.append(param)
            
            st.write(f"✅ **Covered Parameters ({len(covered_params)}):** {', '.join(covered_params[:3])}{'...' if len(covered_params) > 3 else ''}")
            if uncovered_params:
                st.write(f"⚠️ **Needs Assessment ({len(uncovered_params)}):** {', '.join(uncovered_params[:2])}{'...' if len(uncovered_params) > 2 else ''}")

    # --- MAIN APP SECTION ---
    with st.container():
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
    X_input = scaler.transform(input_window[numeric_columns].values)
    X_input = X_input.reshape(1, SEQ_LEN, -1)

    prediction = model.predict(X_input)
    prediction_reshaped = prediction.reshape(PRED_LEN, X_input.shape[2])
    prediction_orig = scaler.inverse_transform(prediction_reshaped)

    # Prepare prediction dataframe
    future_dates = pd.date_range(input_window['Date'].iloc[-1] + pd.Timedelta(days=1), 
                                periods=PRED_LEN, 
                                freq='D')
    pred_df = pd.DataFrame(prediction_orig, 
                          columns=numeric_columns, 
                          index=future_dates)

    # --- PARAMETER SELECTION ---
    param = st.selectbox('Select Parameter to Analyze', 
                        pred_df.columns,
                        index=0,
                        help="Choose which water quality parameter to display and analyze")

    # Show parameter info
    condition, _ = get_parameter_condition(param, pred_df[param].iloc[0])
    st.info(f"**{param}** - First prediction condition: **{condition}**")

    # --- CONDITION CHARTS SECTION ---
    st.subheader('🚦 Water Quality Condition Analysis', divider='blue')
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Selected Parameter", "🌐 All Parameters Overview", "📋 Summary Table", "📈 Trend Analysis"])
    
    with tab1:
        st.markdown(f"### {param} - Condition Forecast")
        
        # Create condition chart for selected parameter
        condition_fig = create_condition_chart(param, pred_df[param].values, future_dates)
        st.plotly_chart(condition_fig, use_container_width=True)
        
        # Show individual day conditions
        st.markdown("#### Daily Condition Breakdown")
        cols = st.columns(5)
        for i, (date, value) in enumerate(zip(future_dates, pred_df[param].values)):
            condition, color = get_parameter_condition(param, value)
            with cols[i]:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border-radius: 10px; background: {color}; color: white; margin: 5px;">
                    <strong>Day {i+1}</strong><br>
                    {date.strftime('%m/%d')}<br>
                    <strong>{condition}</strong><br>
                    {value:.2f}
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### All Parameters - Condition Overview")
        
        # Create overview chart
        overview_fig = create_all_parameters_condition_overview(pred_df, future_dates)
        st.plotly_chart(overview_fig, use_container_width=True)
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🟢 **Good**: Parameter values are within optimal range for safe water
        - 🟡 **Moderate**: Parameter values need attention but are acceptable
        - 🔴 **Bad**: Parameter values exceed safe limits and require immediate action
        - ⚪ **Requires Assessment**: Parameter needs expert evaluation (thresholds not defined)
        """)
    
    with tab3:
        st.markdown("### Condition Summary Table")
        
        # Create and display summary table
        summary_df = create_condition_summary_table(pred_df, future_dates)
        
        if not summary_df.empty:
            # Style the dataframe
            styled_summary = summary_df.style.format({
                'Avg Value': '{:.2f}',
                'Good Days': '{:.0f}',
                'Moderate Days': '{:.0f}',
                'Bad Days': '{:.0f}',
                'Assessment Days': '{:.0f}'
            }).background_gradient(subset=['Good Days'], cmap='Greens')\
              .background_gradient(subset=['Moderate Days'], cmap='Oranges')\
              .background_gradient(subset=['Bad Days'], cmap='Reds')
            
            st.dataframe(styled_summary, use_container_width=True)
            
            # Quick statistics
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
