import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Water Quality Forecasting",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌊 Water Quality Forecasting System")
st.markdown("""
This application predicts water quality parameters for the next **5 days** using the last **10 days** of historical data.
Upload your trained model and CSV dataset to get started.
""")

# Sidebar for file uploads
st.sidebar.header("📁 Upload Files")

# Model upload
model_file = st.sidebar.file_uploader(
    "Upload Trained Model (.h5)",
    type=['h5'],
    help="Upload the trained water quality forecasting model"
)

# CSV upload
csv_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=['csv'],
    help="Upload the water quality dataset with Date column"
)

# Configuration parameters
st.sidebar.header("⚙️ Configuration")
SEQ_LEN = st.sidebar.number_input("Sequence Length (Input Days)", value=10, min_value=5, max_value=30)
PRED_LEN = st.sidebar.number_input("Prediction Length (Output Days)", value=5, min_value=1, max_value=10)

def preprocess_data(df):
    """Preprocess the uploaded CSV data"""
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Drop Quality column if exists
    if 'Quality' in df.columns:
        df = df.drop(columns=['Quality'])
    
    # Handle missing values
    df = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    return df

def create_sequences_for_prediction(data, seq_length=10):
    """Create sequences from the last seq_length days for prediction"""
    if len(data) < seq_length:
        st.error(f"Not enough data! Need at least {seq_length} days of data.")
        return None
    
    # Take the last seq_length days
    last_sequence = data[-seq_length:]
    return np.array([last_sequence])

def inverse_transform_predictions(predictions, scaler, original_shape):
    """Inverse transform predictions to original scale"""
    # Reshape predictions to 2D for inverse transform
    pred_reshaped = predictions.reshape(-1, original_shape)
    pred_orig = scaler.inverse_transform(pred_reshaped)
    # Reshape back to original prediction shape
    return pred_orig.reshape(predictions.shape)

def plot_predictions(historical_data, predictions, feature_names, dates_pred):
    """Create interactive plots for predictions"""
    
    # Select key parameters to display (first 6 parameters)
    n_features_to_show = min(6, len(feature_names))
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=feature_names[:n_features_to_show],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # Get last 30 days of historical data for context
    hist_last_30 = historical_data[-30:]
    hist_dates = pd.date_range(end=dates_pred[0] - timedelta(days=1), periods=len(hist_last_30))
    
    for i in range(n_features_to_show):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=hist_dates,
                y=hist_last_30[:, i],
                mode='lines+markers',
                name=f'Historical {feature_names[i]}',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=row, col=col
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=dates_pred,
                y=predictions[0, :, i],
                mode='lines+markers',
                name=f'Predicted {feature_names[i]}',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Water Quality Predictions - Next 5 Days",
        height=800,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

# Main application logic
if model_file is not None and csv_file is not None:
    try:
        # Load model
        with st.spinner("Loading model..."):
            model = tf.keras.models.load_model(model_file)
        st.success("✅ Model loaded successfully!")
        
        # Load and preprocess data
        with st.spinner("Processing data..."):
            df = pd.read_csv(csv_file, parse_dates=['Date'], dayfirst=True)
            df_processed = preprocess_data(df.copy())
            
        st.success("✅ Data loaded and preprocessed successfully!")
        
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df_processed))
        with col2:
            st.metric("Date Range", f"{df_processed['Date'].min().strftime('%Y-%m-%d')} to {df_processed['Date'].max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("Parameters", len(df_processed.columns) - 1)  # Exclude Date column
        
        # Feature scaling
        feature_columns = [col for col in df_processed.columns if col != 'Date']
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df_processed[feature_columns])
        
        # Create sequences for prediction
        input_sequence = create_sequences_for_prediction(scaled_features, SEQ_LEN)
        
        if input_sequence is not None:
            # Make prediction
            with st.spinner("Making predictions..."):
                predictions_scaled = model.predict(input_sequence)
                
                # Inverse transform predictions
                predictions_orig = inverse_transform_predictions(
                    predictions_scaled, scaler, len(feature_columns)
                )
            
            st.success("✅ Predictions generated successfully!")
            
            # Generate prediction dates
            last_date = df_processed['Date'].max()
            pred_dates = [last_date + timedelta(days=i+1) for i in range(PRED_LEN)]
            
            # Display predictions in a table
            st.subheader("📊 Prediction Results")
            
            # Create prediction DataFrame
            pred_df = pd.DataFrame(
                predictions_orig[0],
                columns=feature_columns,
                index=[f"Day {i+1} ({date.strftime('%Y-%m-%d')})" for i, date in enumerate(pred_dates)]
            )
            
            # Round predictions for better display
            pred_df = pred_df.round(3)
            
            # Display table
            st.dataframe(pred_df.style.highlight_max(axis=0, color='lightcoral').highlight_min(axis=0, color='lightblue'))
            
            # Download predictions
            csv_download = pred_df.to_csv()
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_download,
                file_name=f"water_quality_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Visualizations
            st.subheader("📈 Prediction Visualizations")
            
            # Interactive plot
            historical_data_orig = scaler.inverse_transform(scaled_features)
            fig = plot_predictions(historical_data_orig, predictions_orig, feature_columns, pred_dates)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("📋 Prediction Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Parameter Trends (5-day average vs last day)**")
                summary_data = []
                last_day_values = historical_data_orig[-1]
                pred_avg_values = np.mean(predictions_orig[0], axis=0)
                
                for i, param in enumerate(feature_columns):
                    trend = "📈" if pred_avg_values[i] > last_day_values[i] else "📉"
                    change_pct = ((pred_avg_values[i] - last_day_values[i]) / last_day_values[i]) * 100
                    summary_data.append({
                        "Parameter": param,
                        "Trend": trend,
                        "Change (%)": f"{change_pct:.2f}%"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            
            with col2:
                st.write("**Prediction Confidence Metrics**")
                # Calculate some basic statistics
                pred_std = np.std(predictions_orig[0], axis=0)
                pred_mean = np.mean(predictions_orig[0], axis=0)
                cv = (pred_std / pred_mean) * 100  # Coefficient of variation
                
                confidence_data = []
                for i, param in enumerate(feature_columns):
                    confidence_level = "High" if cv[i] < 10 else "Medium" if cv[i] < 25 else "Low"
                    confidence_data.append({
                        "Parameter": param,
                        "Variability (CV%)": f"{cv[i]:.2f}%",
                        "Confidence": confidence_level
                    })
                
                confidence_df = pd.DataFrame(confidence_data)
                st.dataframe(confidence_df, use_container_width=True)
            
            # Model information
            with st.expander("🔍 Model Information"):
                st.write("**Model Architecture:**")
                model_summary = []
                for i, layer in enumerate(model.layers):
                    model_summary.append({
                        "Layer": i+1,
                        "Type": layer.__class__.__name__,
                        "Output Shape": str(layer.output_shape),
                        "Parameters": layer.count_params()
                    })
                
                model_df = pd.DataFrame(model_summary)
                st.dataframe(model_df, use_container_width=True)
                
                st.write(f"**Total Parameters:** {model.count_params():,}")
                st.write(f"**Input Shape:** {model.input_shape}")
                st.write(f"**Output Shape:** {model.output_shape}")
        
        else:
            st.error("❌ Unable to create input sequences. Please check your data.")
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.write("Please check your files and try again.")

else:
    # Instructions when files are not uploaded
    st.info("👆 Please upload both the trained model (.h5) and CSV dataset to start making predictions.")
    
    st.subheader("📋 Instructions")
    st.markdown("""
    1. **Upload your trained model**: The model should be saved in .h5 format
    2. **Upload your CSV dataset**: Should contain:
        - A 'Date' column with date information
        - Water quality parameter columns (pH, Temperature, etc.)
        - At least 10 days of recent data
    3. **Configure parameters**: Adjust sequence length if needed
    4. **View predictions**: Get 5-day ahead forecasts with visualizations
    
    **Expected CSV Format:**
    ```
    Date,pH,Temperature,Dissolved_Oxygen,Conductivity,BOD,COD,Nitrate,Phosphate,Fecal_Coliform,Total_Coliform,WQI
    2024-01-01,7.2,25.5,8.1,250,2.5,15,1.2,0.8,100,500,75.2
    2024-01-02,7.1,26.0,7.9,245,2.7,16,1.3,0.9,120,520,73.5
    ...
    ```
    """)
    
    # Sample visualization placeholder - no mock data
    st.subheader("📊 Sample Output Preview")
    st.info("Upload your model and dataset to see prediction charts and results here.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666;'>
        🌊 Water Quality Forecasting System | Built with Streamlit & TensorFlow
    </div>
    """, 
    unsafe_allow_html=True
)
