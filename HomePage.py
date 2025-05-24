import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings
import tempfile
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AquaVision AI - Bhagalpur Water Quality Forecasting",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
.status-good { color: #28a745; font-weight: bold; }
.status-medium { color: #ffc107; font-weight: bold; }
.status-high { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">🌊 AquaVision AI - Bhagalpur Water Quality Forecasting</h1>', unsafe_allow_html=True)

def load_csv_data(uploaded_file):
    """Load and preprocess the water quality data"""
    try:
        # Load the uploaded dataset
        df = pd.read_csv(uploaded_file, parse_dates=['Date'], dayfirst=True)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Drop Quality column if present
        if 'Quality' in df.columns:
            df = df.drop(columns=['Quality'])
        
        # Handle missing values
        df = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def load_tensorflow_model(uploaded_model):
    """Load the trained TensorFlow model with error handling"""
    try:
        # Create a temporary file to save the uploaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file.write(uploaded_model.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Try loading with different methods
            model = tf.keras.models.load_model(tmp_file_path, compile=False)
            # Recompile the model
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        except Exception as e1:
            try:
                # Alternative loading method
                model = tf.keras.models.load_model(tmp_file_path, custom_objects=None, compile=False)
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            except Exception as e2:
                st.error(f"Model loading failed with both methods:")
                st.error(f"Method 1: {str(e1)}")
                st.error(f"Method 2: {str(e2)}")
                return None
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
        return model
    except Exception as e:
        st.error(f"Error processing model file: {str(e)}")
        return None

def create_sequences(data, seq_length=10):
    """Create sequences for prediction"""
    if len(data) < seq_length:
        return None
    return data[-seq_length:].reshape(1, seq_length, data.shape[1])

def get_parameter_status(value, param_name):
    """Determine parameter status based on water quality standards"""
    thresholds = {
        'pH': {'good': (6.5, 8.5), 'medium': (6.0, 9.0)},
        'DO': {'good': (5, float('inf')), 'medium': (3, 5)},
        'BOD': {'good': (0, 3), 'medium': (3, 6)},
        'COD': {'good': (0, 20), 'medium': (20, 40)},
        'TSS': {'good': (0, 30), 'medium': (30, 100)},
        'TDS': {'good': (0, 500), 'medium': (500, 1000)},
        'Nitrate': {'good': (0, 10), 'medium': (10, 45)},
        'Phosphate': {'good': (0, 1), 'medium': (1, 5)},
        'Turbidity': {'good': (0, 5), 'medium': (5, 25)},
        'Temperature': {'good': (15, 30), 'medium': (10, 35)},
        'Conductivity': {'good': (0, 400), 'medium': (400, 1000)},
        'WQI': {'good': (75, 100), 'medium': (50, 75)}
    }
    
    if param_name not in thresholds:
        return "UNKNOWN", "⚪"
    
    good_range = thresholds[param_name]['good']
    medium_range = thresholds[param_name]['medium']
    
    if good_range[0] <= value <= good_range[1]:
        return "GOOD", "🟢"
    elif medium_range[0] <= value <= medium_range[1]:
        return "MEDIUM", "🟡"
    else:
        return "HIGH", "🔴"

def predict_next_days(model, data, scaler, last_sequence, pred_days=5):
    """Predict next 5 days using the model"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(pred_days):
        # Predict next day
        pred = model.predict(current_sequence, verbose=0)
        
        # Handle different output shapes
        if pred.ndim == 3:
            next_day = pred[0, 0, :]  # Get first day of prediction
        elif pred.ndim == 2:
            next_day = pred[0, :]
        else:
            next_day = pred.flatten()
        
        predictions.append(next_day)
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, :] = next_day
    
    # Convert predictions to original scale
    predictions = np.array(predictions)
    
    # Ensure predictions have the right shape for inverse transform
    if predictions.shape[1] != scaler.n_features_in_:
        st.error(f"Prediction shape mismatch: got {predictions.shape[1]}, expected {scaler.n_features_in_}")
        return None
    
    predictions_orig = scaler.inverse_transform(predictions)
    
    return predictions_orig

# Main application
def main():
    # File uploaders moved outside of cached functions
    st.sidebar.title("📁 File Upload")
    
    # Data file uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Bhagalpur.csv", 
        type=['csv'],
        help="Please upload your Bhagalpur water quality dataset"
    )
    
    # Model file uploader
    model_file = st.sidebar.file_uploader(
        "Upload Model (.h5)", 
        type=['h5'],
        help="Upload your trained model file"
    )
    
    # Check if files are uploaded
    if uploaded_file is None:
        st.warning("⚠️ Please upload your Bhagalpur.csv file to proceed.")
        st.info("👆 Use the sidebar to upload your data file.")
        return
    
    if model_file is None:
        st.warning("⚠️ Please upload your trained model file to proceed.")
        st.info("👆 Use the sidebar to upload your model file.")
        return
    
    # Load data
    with st.spinner("📊 Loading dataset..."):
        df = load_csv_data(uploaded_file)
    
    if df is None or len(df) < 15:
        st.error("❌ Insufficient data for forecasting. Please upload a valid dataset.")
        return
    
    st.success(f"✅ Dataset loaded successfully! {len(df)} records found.")
    st.info(f"📅 Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Load model
    with st.spinner("🤖 Loading model..."):
        model = load_tensorflow_model(model_file)
    
    if model is None:
        st.error("❌ Failed to load the model. Please check the model file.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar controls
    st.sidebar.title("🎛️ Control Panel")
    
    # Data preprocessing
    scaler = MinMaxScaler()
    feature_cols = [col for col in df.columns if col != 'Date']
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # Get last 10 days for prediction
    last_10_days = scaled_data[-10:]
    last_sequence = create_sequences(scaled_data, seq_length=10)
    
    if last_sequence is None:
        st.error("❌ Not enough data for prediction. Need at least 10 days of data.")
        return
    
    # Make predictions
    with st.spinner("🔮 Generating 5-day forecast..."):
        predictions = predict_next_days(model, scaled_data, scaler, last_sequence)
    
    if predictions is None:
        st.error("❌ Failed to generate predictions. Please check your model and data compatibility.")
        return
    
    # Create prediction dates
    last_date = df['Date'].iloc[-1]
    pred_dates = [last_date + timedelta(days=i+1) for i in range(5)]
    
    # Main dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Water Quality Forecast - Next 5 Days")
        
        # Interactive plot
        fig = make_subplots(
            rows=3, cols=4,
            subplot_titles=feature_cols,
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # Plot historical data (last 10 days) and predictions
        historical_data = df.iloc[-10:][feature_cols].values
        historical_dates = df.iloc[-10:]['Date'].tolist()
        
        colors = px.colors.qualitative.Set3
        
        for i, param in enumerate(feature_cols):
            row = i // 4 + 1
            col = i % 4 + 1
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=historical_dates,
                    y=historical_data[:, i],
                    mode='lines+markers',
                    name=f'{param} (Historical)',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Predictions
            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=predictions[:, i],
                    mode='lines+markers',
                    name=f'{param} (Predicted)',
                    line=dict(color=colors[i % len(colors)], dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add vertical line to separate historical and predicted
            fig.add_vline(
                x=last_date,
                line=dict(color='red', width=2, dash='dash'),
                row=row, col=col
            )
        
        fig.update_layout(height=800, title="Water Quality Parameters - 10 Days Historical + 5 Days Forecast")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Location")
        
        # Bhagalpur coordinates
        bhagalpur_coords = [25.2425, 86.9842]
        
        # Create a simple map
        map_data = pd.DataFrame({
            'lat': [bhagalpur_coords[0]],
            'lon': [bhagalpur_coords[1]]
        })
        
        st.map(map_data, zoom=10)
        
        st.markdown("""
        **📍 Bhagalpur, Bihar**
        - Latitude: 25.2425° N
        - Longitude: 86.9842° E
        - River: Ganges
        """)
    
    # Parameter status dashboard
    st.subheader("🎯 Current Parameter Status")
    
    # Get latest actual data
    latest_data = df.iloc[-1][feature_cols].values
    
    # Create status cards
    cols = st.columns(4)
    for i, param in enumerate(feature_cols):
        status, emoji = get_parameter_status(latest_data[i], param)
        
        with cols[i % 4]:
            if status == "GOOD":
                st.success(f"{emoji} **{param}**\n\n{latest_data[i]:.2f}\n\n{status}")
            elif status == "MEDIUM":
                st.warning(f"{emoji} **{param}**\n\n{latest_data[i]:.2f}\n\n{status}")
            else:
                st.error(f"{emoji} **{param}**\n\n{latest_data[i]:.2f}\n\n{status}")
    
    # Predicted parameter status
    st.subheader("🔮 Predicted Status (Day 1)")
    
    pred_cols = st.columns(4)
    for i, param in enumerate(feature_cols):
        status, emoji = get_parameter_status(predictions[0, i], param)
        
        with pred_cols[i % 4]:
            if status == "GOOD":
                st.success(f"{emoji} **{param}**\n\n{predictions[0, i]:.2f}\n\n{status}")
            elif status == "MEDIUM":
                st.warning(f"{emoji} **{param}**\n\n{predictions[0, i]:.2f}\n\n{status}")
            else:
                st.error(f"{emoji} **{param}**\n\n{predictions[0, i]:.2f}\n\n{status}")
    
    # Data table
    if st.sidebar.checkbox("📋 Show Detailed Predictions"):
        st.subheader("📊 Detailed 5-Day Forecast")
        
        pred_df = pd.DataFrame(predictions, columns=feature_cols)
        pred_df['Date'] = pred_dates
        pred_df = pred_df[['Date'] + feature_cols]
        
        st.dataframe(pred_df.round(2), use_container_width=True)
    
    # Model performance metrics (if available)
    if st.sidebar.checkbox("📈 Model Information"):
        st.subheader("🤖 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", "LSTM Neural Network")
        with col2:
            st.metric("Sequence Length", "10 days")
        with col3:
            st.metric("Prediction Horizon", "5 days")
        
        st.info("💡 The model uses the past 10 days of water quality data to predict the next 5 days.")
        
        # Display model summary if possible
        try:
            st.text("Model Architecture:")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
        except:
            st.info("Model architecture details not available")
    
    # Download predictions
    if st.sidebar.button("💾 Download Predictions"):
        pred_df = pd.DataFrame(predictions, columns=feature_cols)
        pred_df['Date'] = pred_dates
        pred_df = pred_df[['Date'] + feature_cols]
        
        csv = pred_df.to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"bhagalpur_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
