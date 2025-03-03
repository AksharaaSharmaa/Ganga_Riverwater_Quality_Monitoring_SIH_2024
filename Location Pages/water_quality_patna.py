import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import MinMaxScaler
import altair as alt
import matplotlib.dates as mdates
import folium
from streamlit_folium import folium_static
import google.generativeai as genai
from matplotlib import gridspec
from matplotlib.patches import Arc
import os

# Load API keys from environment variables
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Check if all required keys are available
if not GOOGLE_MAPS_API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY not found. Make sure it is set in your environment variables.")
if not WEATHER_API_KEY:
    raise ValueError("WEATHER_API_KEY not found. Make sure it is set in your environment variables.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Make sure it is set in your environment variables.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Updated Locations dictionary
LOCATIONS = {
    'GANGA AT GULABI GHAT, PATNA': {
        'file_path': r"Datasets\Patna.csv",
        'lat': 25.620356,
        'lon': 85.179995
    }  
}


def get_time_based_greeting():
    """Determine greeting based on current time with enhanced details."""
    from datetime import datetime
    current_hour = datetime.now().hour
    
    # Expanded greeting logic with more nuanced time-based styling
    if 5 <= current_hour < 12:
        return {
            "greeting": "Good Morning",
            "bg_color": "#FFF4E0",
            "accent_color": "#FFD700", 
            "text_color": "#333333",
            "icon": "☀️",
            "message": "Rise and shine! A brand new day awaits your brilliance.",
            "gradient_start": "#FFFFFF",
            "gradient_end": "#FFE5B4"
        }
    elif 12 <= current_hour < 17:
        return {
            "greeting": "Good Afternoon", 
            "bg_color": "#E6F3FF",
            "accent_color": "#87CEEB",
            "text_color": "#2C3E50", 
            "icon": "☕",
            "message": "Halfway through the day. Keep pushing forward!",
            "gradient_start": "#F0F8FF",
            "gradient_end": "#87CEEB"
        }
    elif 17 <= current_hour < 22:
        return {
            "greeting": "Good Evening",
            "bg_color": "#E6E6FA", 
            "accent_color": "#DDA0DD",
            "text_color": "#4A4A4A",
            "icon": "🌆",
            "message": "Time to unwind and reflect on your day's achievements.",
            "gradient_start": "#F0E6FF", 
            "gradient_end": "#DDA0DD"
        }
    else:
        return {
            "greeting": "Good Night", 
            "bg_color": "#191970", 
            "accent_color": "#4169E1",
            "text_color": "#FFFFFF",
            "icon": "🌙",
            "message": "Rest well. Tomorrow brings new opportunities.",
            "gradient_start": "#000080", 
            "gradient_end": "#191970"
        }

def create_custom_css():
    """Create custom CSS for improved styling and hover effects."""
    return f"""
    <style>
    .greeting-card {{
        background: linear-gradient(135deg, 
            {st.session_state.time_details['gradient_start']} 0%, 
            {st.session_state.time_details['gradient_end']} 100%);
        border-radius: 10px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
        color: {st.session_state.time_details['text_color']};
        position: relative;
        overflow: hidden;
        margin-bottom: 20px; /* Added space after KPI card */
    }}
    
    .greeting-card:hover {{
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }}
    
    .greeting-icon {{
        font-size: 3rem;
        margin-bottom: 10px;
        animation: float 2s ease-in-out infinite;
    }}
    
    .greeting-title {{
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 10px;
        color: {st.session_state.time_details['accent_color']};
    }}
    
    .greeting-date {{
        font-size: 1rem;
        color: {st.session_state.time_details['text_color']};
        margin-bottom: 10px;
    }}
    
    .greeting-message {{
        font-size: 0.9rem;
        font-style: italic;
        opacity: 0.8;
        margin-top: 10px;
    }}
    
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
        100% {{ transform: translateY(0px); }}
    }}
    </style>
    """

def create_greeting_card():
    """Create an enhanced, interactive greeting card using Streamlit."""
    # Retrieve time-based greeting details
    st.session_state.time_details = get_time_based_greeting()
    
    # Inject custom CSS
    st.markdown(create_custom_css(), unsafe_allow_html=True)
    
    # Create card container with enhanced styling
    st.markdown(f'''
    <div class="greeting-card">
        <div class="greeting-icon">{st.session_state.time_details['icon']}</div>
        <div class="greeting-title">{st.session_state.time_details['greeting']}</div>
        <div class="greeting-date">{datetime.now().strftime('%A, %B %d, %Y')}</div>
    ''', unsafe_allow_html=True)

def create_satellite_map(latitude, longitude):
    """
    Create a Folium map with satellite view for a given location
    """
    # Create a map centered on the location with satellite view
    m = folium.Map(
        location=[latitude, longitude], 
        zoom_start=13,
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&hl=en',
        attr='Google Satellite'
    )
    
    # Add a marker for the exact location
    folium.Marker(
        [latitude, longitude],
        popup='Monitoring Location',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    return m

def parse_date(date_str):
    """
    Flexible date parsing function to handle different date formats
    """
    date_formats = [
        '%d-%m-%Y',   # Day-Month-Year (13-01-2020)
        '%m-%d-%Y',   # Month-Day-Year (01-13-2020)
        '%Y-%m-%d',   # Year-Month-Day (2020-01-13)
        '%d/%m/%Y',   # Day/Month/Year (13/01/2020)
        '%m/%d/%Y',   # Month/Day/Year (01/13/2020)
    ]
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    # If no format works, raise an error
    raise ValueError(f"Unable to parse date: {date_str}")

def prepare_input_data(historical_data, weather_forecast, parameter):
    """
    Prepare input data for model prediction with fixed sequence length
    """
    # Ensure 'Date' column is in datetime format
    historical_data['Date'] = historical_data['Date'].apply(parse_date)
    historical_data = historical_data.dropna(subset=['Date'])
    historical_data.set_index('Date', inplace=True)
    
    # Get the last 10 days of data
    last_10_days = historical_data[parameter].tail(10)
    
    # Ensure we have exactly 5 forecast points
    if len(weather_forecast) < 5:
        # Pad with the last forecast values if not enough
        while len(weather_forecast) < 5:
            last_forecast = weather_forecast[-1]
            weather_forecast.append({
                'date': last_forecast['date'] + timedelta(days=1),
                'temperature': last_forecast['temperature'],
                'rainfall': last_forecast['rainfall']
            })
    elif len(weather_forecast) > 5:
        # Trim to first 5 forecasts
        weather_forecast = weather_forecast[:5]
    
    # Prepare exogenous variables
    temps = [w['temperature'] for w in weather_forecast]
    rainfalls = [w['rainfall'] for w in weather_forecast]
    
    # Scale the time series data
    param_scaler = MinMaxScaler()
    scaled_data = param_scaler.fit_transform(last_10_days.values.reshape(-1, 1))
    
    # Scale exogenous variables separately
    temp_scaler = MinMaxScaler()
    rainfall_scaler = MinMaxScaler()
    
    scaled_temps = temp_scaler.fit_transform(np.array(temps).reshape(-1, 1))
    scaled_rainfalls = rainfall_scaler.fit_transform(np.array(rainfalls).reshape(-1, 1))
    
    # Combine scaled exogenous variables
    scaled_exogenous = np.column_stack([scaled_temps.flatten(), scaled_rainfalls.flatten()])
    
    # Reshape sequences
    X = scaled_data.reshape(1, 10, 1)  # Time series input
    X_exo = scaled_exogenous.reshape(1, 5, 2)  # Exogenous variables
    
    return X, X_exo, param_scaler, last_10_days, temp_scaler, rainfall_scaler

def fetch_weather_forecast(location, start_date):
    location_coords = {
        'GANGA AT GULABI GHAT, PATNA': {'lat': 25.620356,'lon': 85.179995}
    }
    
    try:
        coords = location_coords[location]
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={coords['lat']}&lon={coords['lon']}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        # Group forecasts by date and calculate max temperature
        forecasts_by_date = {}
        for forecast in data['list']:
            forecast_date = datetime.fromtimestamp(forecast['dt']).date()
            
            if forecast_date >= start_date.date():
                temp = forecast['main']['temp']
                rainfall = forecast.get('rain', {}).get('3h', 0) or 0

                if forecast_date not in forecasts_by_date:
                    forecasts_by_date[forecast_date] = {
                        'date': datetime.combine(forecast_date, datetime.min.time()),
                        'temperature': temp,
                        'rainfall': rainfall
                    }
                else:
                    # Update max temperature and add rainfall
                    forecasts_by_date[forecast_date]['temperature'] = max(forecasts_by_date[forecast_date]['temperature'], temp)
                    forecasts_by_date[forecast_date]['rainfall'] += rainfall

        # Convert to list and limit to 5 forecasts
        forecasts = list(forecasts_by_date.values())
        
        # Ensure exactly 5 forecasts
        while len(forecasts) < 5:
            last_forecast = forecasts[-1]
            forecasts.append({
                'date': last_forecast['date'] + timedelta(days=1),
                'temperature': last_forecast['temperature'],
                'rainfall': last_forecast['rainfall']
            })
        
        return forecasts[:5]
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        # Generate dummy forecasts if API fails
        dummy_forecasts = [
            {
                'date': start_date + timedelta(days=i+1),
                'temperature': 25.0,  # Default temperature
                'rainfall': 0.0  # Default rainfall
            } for i in range(5)
        ]
        return dummy_forecasts


def create_altair_forecast_plot(historical_data, forecast_data, parameter):
    """
    Create an Altair plot showing historical and forecasted data
    """
    # Convert index to datetime if it's not already
    if not isinstance(historical_data.index, pd.DatetimeIndex):
        historical_data.index = pd.to_datetime(historical_data.index)
    
    # Prepare historical data
    historical_df = pd.DataFrame({
        'Date': historical_data.index,
        'Value': historical_data.values,
        'Type': 'Historical'
    })
    
    # Prepare forecast data
    forecast_dates = [historical_data.index[-1] + timedelta(days=i+1) for i in range(len(forecast_data))]
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Value': forecast_data,
        'Type': 'Forecast'
    })
    
    # Combine historical and forecast data
    combined_df = pd.concat([historical_df, forecast_df])
    
    # Create the Altair chart
    chart = alt.Chart(combined_df).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Value:Q', title=parameter),
        color=alt.Color('Type:N', 
            scale=alt.Scale(domain=['Historical', 'Forecast'], 
                            range=['steelblue', 'red']),
            legend=alt.Legend(title='Data Type')
        ),
        tooltip=['Date:T', 'Value:Q', 'Type:N']
    ).properties(
        title=f'{parameter} - Historical and Forecast',
        width=700,
        height=400
    ).interactive()
    
    return chart

def load_model_for_parameter(parameter):
    """
    Load the pre-trained model for a specific water quality parameter.
    """
    parameter_model_paths = {
        "Biochemical Oxygen Demand": r"models\Biochemical_Oxygen_Demand_water_quality_lstm_model.keras",
        "Dissolved Oxygen": r"models\Dissolved_Oxygen_water_quality_lstm_model.keras",
        "pH": r"models\pH_water_quality_lstm_model.keras",
        "Turbidity": r"models\Turbidity_water_quality_lstm_model.keras",
        "Nitrate": r"models\Nitrate_water_quality_lstm_model.keras",
        "Fecal Coliform": r"models\Fecal_Coliform_water_quality_lstm_model.keras",
        "Fecal Streptococci": r"models\Fecal_Streptococci_water_quality_lstm_model.keras",
        "Total Coliform": r"models\Total_Coliform_water_quality_lstm_model.keras",
        "Conductivity": r"models\Conductivity_water_quality_lstm_model.keras"
    }
    model_path = parameter_model_paths.get(parameter)
    
    if model_path and os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error(f"Model for {parameter} not found.")
        return None


def create_altair_historical_plot(df, parameter):
    """
    Create an Altair plot of historical data
    """
    # Create the base chart
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y(f'{parameter}:Q', title=parameter),
        tooltip=['Date:T', f'{parameter}:Q']
    ).properties(
        title=f'Historical {parameter} Data',
        width=700,
        height=400
    ).interactive()
    
    return chart

def generate_gemini_water_quality_report(parameter, forecasted_values, forecast_dates, historical_data):
    """
    Generate a detailed water quality report using Gemini API
    
    Args:
    parameter (str): Water quality parameter
    forecasted_values (list): Predicted values
    forecast_dates (list): Corresponding dates
    historical_data (pd.DataFrame): Historical water quality data
    
    Returns:
    str: Detailed water quality report from Gemini
    """
    # Prepare input context
    historical_stats = f"""
    Historical Data Statistics for {parameter}:
    - Mean: {historical_data[parameter].mean():.4f}
    - Standard Deviation: {historical_data[parameter].std():.4f}
    - Minimum: {historical_data[parameter].min():.4f}
    - Maximum: {historical_data[parameter].max():.4f}
    """
    
    # Format forecasted data
    forecast_details = "\n".join([
        f"Date: {date.strftime('%Y-%m-%d')}, Predicted Value: {value:.4f}"
        for date, value in zip(forecast_dates, forecasted_values)
    ])
    
    # Construct prompt for Gemini
    prompt = f"""
    Provide a comprehensive water quality report for {parameter} with the following details and dont give in table format:

    {historical_stats}

    Forecasted Values:
    {forecast_details}

    For each forecast date, please analyze:
    1. Potential water quality implications
    2. Risk assessment
    3. Recommended actions

    Format the report with clear headings and provide actionable insights.
    """
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate report
        response = model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        return f"Error generating report: {str(e)}"

# Integration in main Streamlit function
def append_gemini_report_to_streamlit(parameter, forecasted_values, forecast_dates, historical_data):
    """
    Generate and display Gemini-powered water quality report in Streamlit
    """    
    # Generate report
    gemini_report = generate_gemini_water_quality_report(
        parameter, 
        forecasted_values, 
        forecast_dates, 
        historical_data
    )
    
    # Display report
    st.markdown(gemini_report)
    
def get_status_details(value, parameter_data, selected_parameter):
    """
    Determine status based on actual parameter thresholds
    
    Args:
    value (float): Current forecast value
    parameter_data (pd.DataFrame): Historical data
    selected_parameter (str): Current water quality parameter
    
    Returns:
    tuple: (status, color, risk_percentage)
    """
    # Calculate thresholds based on historical data
    mean = parameter_data[selected_parameter].mean()
    std = parameter_data[selected_parameter].std()
    
    # Define thresholds and risk levels
    if value < mean - std:
        return "Low Risk", "green", 30
    elif mean - std <= value < mean + std:
        return "Moderate Risk", "orange", 60
    else:
        return "High Risk", "red", 90


def create_altair_weather_kpi(location, last_date):
    """
    Create an Altair-based Weather KPI visualization using actual weather data
    
    Args:
    location (str): Location name
    last_date (datetime): Last date in the dataset
    
    Returns:
    alt.Chart: Altair chart representing weather KPI
    """
    # Use existing location coordinates from the LOCATIONS dictionary
    location_coords = {
        'GANGA AT GULABI GHAT, PATNA': {'lat': 25.620356,'lon': 85.179995}
    }
    
    try:
        # Fetch current weather data
        coords = location_coords[location]
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={coords['lat']}&lon={coords['lon']}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        weather_data = response.json()
        
        # Prepare data for Altair visualization
        # Add precipitation data (if available)
        precipitation = weather_data.get('rain', {}).get('1h', 0) if 'rain' in weather_data else 0
        
        weather_metrics = pd.DataFrame([
            {
                'Metric': 'Temperature', 
                'Value': weather_data['main']['temp'], 
                'Unit': '°C'
            },
            {
                'Metric': 'Humidity', 
                'Value': weather_data['main']['humidity'], 
                'Unit': '%'
            },
            {
                'Metric': 'Wind Speed', 
                'Value': weather_data['wind']['speed'], 
                'Unit': 'km/h'
            },
            {
                'Metric': 'Precipitation', 
                'Value': precipitation, 
                'Unit': 'mm/h'
            }
        ])
        
        # Color scale for bars
        color_scale = alt.Color('Metric:N', 
            scale=alt.Scale(
                domain=['Temperature', 'Humidity', 'Wind Speed', 'Precipitation'],
                range=['#FF6B6B', '#4ECDC4', '#2575FC', '#8A4FFF']
            )
        )
        
        # Create base chart
        base = alt.Chart(weather_metrics).mark_bar(
            cornerRadius=5
        ).encode(
            x=alt.X('Metric:N', 
                title=None, 
                axis=alt.Axis(labelAngle=0, labelPadding=5)
            ),
            y=alt.Y('Value:Q', 
                title=None,
                axis=alt.Axis(grid=False)
            ),
            color=color_scale,
            tooltip=['Metric', 'Value', 'Unit']
        ).properties(
            title={
                'text': f'Weather Overview - {location}',
                'subtitle': f'As of {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                'color': '#6a11cb',
                'fontSize': 16,
                'fontWeight': 'bold'
            },
            width=400,
            height=250
        )
        
        # Add text labels on top of bars
        text = base.mark_text(
            dy=-10,
            color='black',
            fontWeight='bold'
        ).encode(
            text=alt.Text('Value:Q', format='.1f'),
            color=color_scale
        )
        
        # Combine base chart and text
        chart = base + text
        
        return chart
    
    except Exception as e:
        # Fallback to error handling if API fails
        st.error(f"Could not fetch weather data: {e}")
        
        # Create a default chart with placeholder data
        default_data = pd.DataFrame([
            {'Metric': 'Temperature', 'Value': 13, 'Unit': '°C'},
            {'Metric': 'Humidity', 'Value': 0, 'Unit': '%'},
            {'Metric': 'Wind Speed', 'Value': 3, 'Unit': 'km/h'},
            {'Metric': 'Precipitation', 'Value': 0, 'Unit': 'mm/h'}
        ])
        
        base = alt.Chart(default_data).mark_bar(
            color='lightgray',
            cornerRadius=5
        ).encode(
            x='Metric:N',
            y='Value:Q'
        ).properties(
            title='Weather Data Unavailable',
            width=400,
            height=250
        )
        
        return base


def make_donut(input_response, input_text, parameter, parameter_data, selected_parameter):
    """
    Create an Altair donut chart for water quality forecast visualization
    
    Args:
    input_response (float): The actual forecast value
    input_text (str): Label for the chart
    parameter (str): Water quality parameter for color selection
    parameter_data (pd.DataFrame): Historical dataframe
    selected_parameter (str): Current selected parameter
    
    Returns:
    alt.Chart: Altair donut chart
    """
    # Determine risk level and color
    status, risk_color, risk_percentage = get_status_details(
        input_response, parameter_data, selected_parameter
    )
    
    # Color mapping based on risk level
    color_map = {
        'Low Risk': ['#27AE60', '#12783D'],  # Green
        'Moderate Risk': ['#F39C12', '#875A12'],  # Orange
        'High Risk': ['#E74C3C', '#FAF9F6']  # Red
    }
    
    chart_color = color_map.get(status, color_map['Moderate Risk'])
    
    # Prepare data for donut chart
    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100 - risk_percentage, risk_percentage]
    })
    
    # Background donut (full circle)
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })
    
    # Create base donut chart
    plot = alt.Chart(source).mark_arc(
        innerRadius=45, 
        cornerRadius=25
    ).encode(
        theta="% value",
        color=alt.Color(
            "Topic:N", 
            scale=alt.Scale(
                domain=[input_text, ''],
                range=chart_color
            ),
            legend=None
        )
    ).properties(width=130, height=130)
    
    # Add text overlay - show actual forecast value
    text = alt.Chart(source).mark_text(
        align='center', 
        color=chart_color[0], 
        font="Arial", 
        fontSize=16, 
        fontWeight='bold'
    ).encode(
        text=alt.value(f'{input_response:.2f}')
    ).properties(width=130, height=130)
    
    # Background donut chart
    plot_bg = alt.Chart(source_bg).mark_arc(
        innerRadius=45, 
        cornerRadius=20
    ).encode(
        theta="% value",
        color=alt.Color(
            "Topic:N", 
            scale=alt.Scale(
                domain=[input_text, ''],
                range=chart_color
            ),
            legend=None
        )
    ).properties(width=130, height=130)
    
    # Combine charts
    return plot_bg + plot + text



def enhanced_dashboard_layout():
    """
    Create an aesthetically pleasing dashboard layout with more modern styling
    """
    st.markdown("""
    <style>
    /* Modern, Clean Design */
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    
    /* Glassmorphic Card Design */
    .dashboard-card {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 16px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 30px rgba(31, 38, 135, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(31, 38, 135, 0.15);
    }
    
    /* Gradient Headings */
    .gradient-text {
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
    }
    
    /* Enhanced Tab Design */
    .stTabs > div > div > div {
        background: linear-gradient(135deg, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0.4) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    
    /* Smooth Animations */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated-section {
        animation: slideIn 0.6s ease-out;
    }
    
    /* Enhanced Select Box */
    .stSelectbox > div > div > div {
        background: rgba(255,255,255,0.7) !important;
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

def display_water_quality_alerts(forecast_values, parameter_data):
    """
    Display water quality alerts in Streamlit based on forecast values and historical data
    
    Args:
    forecast_values (dict): Dictionary of parameter names and their forecast values
    parameter_data (pd.DataFrame): Historical parameter data for threshold calculations
    """
    def get_risk_level(value, parameter_name):
        """
        Determine risk level for a specific parameter
        
        Args:
        value (float): Current forecast value
        parameter_name (str): Name of the parameter
        
        Returns:
        str: Risk level ('low', 'moderate', 'high')
        """
        # Calculate thresholds based on historical data
        mean = parameter_data[parameter_name].mean()
        std = parameter_data[parameter_name].std()
        
        # Define risk levels based on standard deviation
        if value < mean - std:
            return 'low'
        elif mean - std <= value < mean + std:
            return 'moderate'
        else:
            return 'high'

    # Configuration for different risk levels
    risk_config = {
        'low': {
            'icon': '✅',
            'color': 'green',
            'title': 'Low Risk',
            'description': 'Within expected range'
        },
        'moderate': {
            'icon': '⚠️',
            'color': 'orange',
            'title': 'Moderate Risk',
            'description': 'Approaching threshold limits'
        },
        'high': {
            'icon': '🚨',
            'color': 'red',
            'title': 'High Risk',
            'description': 'Exceeding safe limits'
        }
    }

    # Track parameters that are not low risk
    risk_parameters = []

    # Check risk for each parameter
    for param_name, value in forecast_values.items():
        risk_level = get_risk_level(value, param_name)
        
        # Only add non-low risk parameters
        if risk_level != 'low':
            risk_parameters.append({
                'name': param_name,
                'value': value,
                'risk_level': risk_level
            })

    # Display alerts
    if not risk_parameters:
        # All parameters are low risk
        st.success("✅ Excellent Water Quality: All parameters are within safe limits.")
    else:
        # Display alerts for risky parameters
        for param in risk_parameters:
            config = risk_config[param['risk_level']]
            
            # Create color-coded alert
            st.markdown(f"""
            <div style="
                background-color: {config['color']}20; 
                border: 2px solid {config['color']}; 
                border-radius: 10px; 
                padding: 15px; 
                margin-bottom: 10px;
            ">
            <h4 style="color: {config['color']}; margin-top: 0;">
                {config['icon']} {config['title']} - {param['name'].replace('_', ' ').title()}
            </h4>
            <p style="margin-bottom: 5px;">{config['description']}</p>
            <p style="font-weight: bold; color: {config['color']};">
                Forecast Value: {param['value']:.2f}
            </p>
            </div>
            """, unsafe_allow_html=True)
    
def main():
    st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    .dashboard-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        padding: 20px;
    }
    .gradient-text {
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .donut-container {
        display: flex;
        flex-direction: row;
        justify-content: center;
        align-items: center;
        gap: 20px;
        width: 100%;
    }
    .donut-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add enhanced styling with sliding and arrows
    st.markdown("""
    <style>
    /* Wrapper for the tab container with arrows */
    .tab-container {
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
    }

    /* Tab list for horizontal scrolling */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        overflow-x: auto;
        scroll-behavior: smooth;
        gap: 15px;
        padding: 10px;
        margin: 0;
        background: linear-gradient(135deg, #f6f8fc 0%, #e9ecf4 100%);
        border-radius: 15px;
        border: 1px solid rgba(102, 102, 102, 0.125);
    }

    /* Hide the scrollbar */
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }

    /* Left and right navigation buttons */
    .scroll-arrow {
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        background-color: white;
        border: 1px solid rgba(102, 102, 102, 0.125);
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        z-index: 1;
    }

    .scroll-arrow.left {
        left: 10px;
    }

    .scroll-arrow.right {
        right: 10px;
    }

    /* Arrow hover effects */
    .scroll-arrow:hover {
        background-color: #f0f0f6;
        box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        transition: all 0.3s ease;
        background-color: white;
        border-radius: 10px;
        padding: 12px 20px;
        font-weight: 600;
        color: #6a11cb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9em;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f0f6;
        transform: scale(1.05);
        box-shadow: 0 3px 6px rgba(0,0,0,0.12);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>

    <div class="tab-container">
        <div class="scroll-arrow left" onclick="scrollTabs(-200)">&#9664;</div>
        <div id="tab-list-container" class="stTabs [data-baseweb='tab-list']"></div>
        <div class="scroll-arrow right" onclick="scrollTabs(200)">&#9654;</div>
    </div>

    <script>
    function scrollTabs(offset) {
        const tabList = document.querySelector('.stTabs [data-baseweb="tab-list"]');
        if (tabList) {
            tabList.scrollBy({ left: offset, behavior: 'smooth' });
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Enhanced CSS Styling
    enhanced_dashboard_layout()

    # Title with Gradient
    st.markdown('<h1 class="gradient-text">GANGA AT GULABI GHAT, PATNA</h1>', unsafe_allow_html=True)

    # Hardcoded Location
    selected_location = "GANGA AT GULABI GHAT, PATNA"
    
    if selected_location:
        # Load location-specific dataset
        df = pd.read_csv(LOCATIONS[selected_location]['file_path'])
        df['Date'] = df['Date'].apply(parse_date)
        
        # Get the last date in the dataset
        last_date = df['Date'].max()

        # Weather KPI Cards
        col1, col2 = st.columns(2)

        with col1:
            create_greeting_card()
            
            # Weather KPI using Altair
            weather_kpi = create_altair_weather_kpi(selected_location, df['Date'].max())
            st.altair_chart(weather_kpi, use_container_width=True)

        with col2:
            # Satellite Map
            location_data = LOCATIONS[selected_location]
            satellite_map = create_satellite_map(location_data['lat'], location_data['lon'])
            folium_static(satellite_map, width=500, height=500)
        
        parameters = [
            param for param in df.columns 
            if param not in ['Date', 'Temperature', 'Rainfall']
        ]

        # Modify parameter names for better readability
        formatted_parameters = [
            param.replace('_', ' ').title() for param in parameters
        ]

        # Create Tabs with formatted parameter names
        tabs = st.tabs(formatted_parameters)
        
        # Iterate through tabs and create content for each parameter
        for idx, parameter in enumerate(parameters):
            with tabs[idx]:
                
                # Prediction and Visualization
                try:
                    # Get the last date in the dataset
                    last_date = df['Date'].max()
                    
                    # Fetch Weather Forecast
                    weather_forecasts = fetch_weather_forecast(selected_location, last_date)
                    
                    # Load Appropriate Model
                    model = load_model_for_parameter(parameter)

                    if model and weather_forecasts:
                        # Prepare the input data for prediction
                        historical_features, exogenous_features, scaler, last_10_days, temp_scaler, rainfall_scaler = prepare_input_data(
                            df, weather_forecasts, parameter
                        )

                        # Predict
                        prediction = model.predict([historical_features, exogenous_features])
                        predicted_values = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
                        forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(predicted_values))]

                        # Create two side-by-side graphs
                        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Past Year Data
                            past_year_data = df[df['Date'] > (last_date - timedelta(days=365))]
                            st.subheader('Past Year Data')
                            historical_chart = create_altair_historical_plot(past_year_data, parameter)
                            st.altair_chart(historical_chart, use_container_width=True)
                        
                        with col2:
                            # 10 Day Historical and 5 Day Forecast
                            st.subheader('10 Day Historical and 5 Day Forecast')
                            forecast_chart = create_altair_forecast_plot(last_10_days, predicted_values, parameter)
                            st.altair_chart(forecast_chart, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Water Quality Alerts
                        forecast_dict = {parameter: predicted_values[-1]}
                        display_water_quality_alerts(forecast_dict, df)

                        # Daily Water Quality Forecast
                        st.subheader("Daily Water Quality Forecast")
                        
                        # Create two columns for donut charts and Gemini report
                        donut_col, report_col = st.columns(2)

                        with donut_col:
                            st.markdown('''
                            <style>
                            .donut-container {
                                display: flex;
                                flex-direction: row;
                                justify-content: center;
                                align-items: center;
                                gap: 20px;
                                width: 100%;
                            }
                            .donut-item {
                                display: flex;
                                flex-direction: column;
                                align-items: center;
                            }
                            </style>
                            <div class="donut-container">
                            ''', unsafe_allow_html=True)

                            # Create donut charts
                            for idx, (date, value) in enumerate(zip(forecast_dates, predicted_values)):
                                donut = make_donut(
                                    input_response=value,  
                                    input_text=date.strftime("%Y-%m-%d"),
                                    parameter=parameter,
                                    parameter_data=df,
                                    selected_parameter=parameter
                                )
                                
                                # Wrap each donut in a div for individual styling
                                st.markdown(f'<div class="donut-item">', unsafe_allow_html=True)
                                st.altair_chart(donut, use_container_width=False)
                                st.markdown(f'<p>{date.strftime("%Y-%m-%d")}</p>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)

                            # Close the container
                            st.markdown('</div>', unsafe_allow_html=True)

                        with report_col:
                            # Generate Gemini AI Report
                            try:
                                # Generate report and display
                                gemini_report = generate_gemini_water_quality_report(
                                    parameter, 
                                    predicted_values, 
                                    forecast_dates, 
                                    df
                                )
                                st.markdown("### AI-Generated Insights")
                                st.markdown(gemini_report)
                            except Exception as e:
                                st.error(f"Could not generate AI insights: {e}")

                except Exception as e:
                    st.error(f"Error processing data for {parameter}: {e}")

if __name__ == "__main__":
    main()
