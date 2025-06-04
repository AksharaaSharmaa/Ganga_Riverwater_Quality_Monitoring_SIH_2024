import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
import google.generativeai as genai
import json

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
    .insight-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1e3a8a;
        margin: 10px 0;
    }
    .ai-response {
        background: #f8fafc;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
SEQ_LEN = 10
PRED_LEN = 5
MODEL_PATH = 'bhagalpur_final_water_quality_forecasting_model.h5'
DATA_PATH = 'Bhagalpur.csv'

# Gemini API Key (Replace with your actual API key)
GEMINI_API_KEY = "AIzaSyAldo6EIJngpc9TRS58sk3JOCC5ib4E858"  # Replace this with your actual Gemini API key

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

    # --- PARAMETER SELECTION ---
    param = st.selectbox('Select Parameter to Analyze', 
                        pred_df.columns,
                        index=0,
                        help="Choose which water quality parameter to display and analyze")

    # --- VISUALIZATION ---
    st.subheader('📈 5-Day Prediction Analysis', divider='blue')

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
