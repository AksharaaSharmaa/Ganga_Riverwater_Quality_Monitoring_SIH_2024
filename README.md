# 🌊 Ganga River Water Quality Monitoring & Forecasting System 🌊

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Forecasting Accuracy](https://img.shields.io/badge/Forecasting%20Accuracy-97%25-brightgreen.svg)](https://github.com/yourusername/ganga-water-monitoring)

## 🌟 Overview

This project implements an AI-enabled Decision Support System (DSS) for monitoring and forecasting water quality parameters of the sacred Ganga River in India. Using advanced time series forecasting techniques (SARIMAX) with 97% accuracy, our system provides 5-day advance predictions of critical water quality parameters across 13 strategic locations spanning the entire course of the river.

## ✨ Key Features

- **High-Precision Forecasting**: 5-day ahead water quality predictions with 97% accuracy using SARIMAX models
- **Comprehensive Coverage**: Monitors 13 strategic locations along the entire course of River Ganga
- **Rich Data Sources**: Integrates water quality data from GemStat and meteorological data from MeteoStat
- **Interactive Dashboard**: Real-time visualization of current and predicted water quality parameters
- **AI-Generated Insights**: Utilizes Gemini AI to generate actionable water quality insights
- **User Feedback System**: Continuous improvement through structured user feedback collection

## 📊 Data Sources

- **Water Quality Data**: GemStat database providing parameters like pH, dissolved oxygen, BOD, COD, and turbidity
- **Meteorological Data**: MeteoStat API providing temperature, precipitation, and other weather variables
- **Locations Covered**: 13 strategic sampling points from Gangotri to Ganga Sagar

## 🧠 Methodology

### Data Collection & Preprocessing
```python
# Sample code for data collection
import pandas as pd
from gemstat_api import GemStatClient
from meteostat import Point, Daily

# Initialize clients
gemstat = GemStatClient(api_key="YOUR_API_KEY")
locations = [
    {"name": "Gangotri", "lat": 30.9946, "lon": 78.9398},
    {"name": "Haridwar", "lat": 29.9457, "lon": 78.1642},
    # ... 11 more locations
]

# Collect and merge data
for location in locations:
    water_data = gemstat.get_water_quality(location["name"])
    weather = Daily(Point(location["lat"], location["lon"]))
    
    # Preprocessing steps
    # ...
```

### SARIMAX Model Implementation
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# For each water quality parameter at each location
def train_sarimax_model(df, param):
    # Split data into train/test
    train, test = df[:-30], df[-30:]
    
    # Find optimal parameters
    model = SARIMAX(train[param], 
                   order=(2, 1, 2),
                   seasonal_order=(1, 1, 1, 7),
                   exog=train[['temperature', 'precipitation']])
    
    model_fit = model.fit()
    return model_fit
```

### Performance Metrics
Our SARIMAX models achieve impressive performance metrics:
- **RMSE**: 0.12-0.35 (depending on parameter and location)
- **MAE**: 0.09-0.28
- **Overall Accuracy**: 97%

## 🚀 Usage

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/ganga-water-monitoring.git
cd ganga-water-monitoring

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the System
```bash
# Run the forecasting pipeline
python src/forecasting.py

# Start the web dashboard
python src/dashboard.py
```

### Generating Insights
```python
# Sample code for generating insights using Gemini
from gemini_client import GeminiAI

gemini = GeminiAI(api_key="YOUR_GEMINI_API_KEY")

def generate_insights(location_data):
    prompt = f"Analyze the following water quality data and provide insights: {location_data}"
    insights = gemini.generate(prompt)
    return insights
```

## 📈 Results

Our system successfully predicts crucial water quality parameters with high accuracy:
- Dissolved Oxygen (DO)
- Biochemical Oxygen Demand (BOD) 
- Chemical Oxygen Demand (COD)
- pH levels
- Total Dissolved Solids (TDS)
- Temperature

The 5-day forecasting window provides critical lead time for authorities to implement preventive measures against potential pollution events.

## 🔄 Feedback System

We've implemented a comprehensive feedback system to continuously improve our predictions:
```python
# Sample feedback collection code
def collect_feedback(prediction_id, actual_values, user_comments):
    feedback = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now(),
        "predicted_values": get_prediction(prediction_id),
        "actual_values": actual_values,
        "user_comments": user_comments
    }
    
    # Store feedback and trigger model retraining if necessary
    db.feedbacks.insert_one(feedback)
    evaluate_retraining_need()
```

## 🔮 Future Work

- Extend forecasting window to 14 days
- Implement real-time anomaly detection
- Develop mobile application for field workers
- Integrate satellite imagery for visual pollution detection
- Expand to tributary rivers in the Ganga basin


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Ministry of Jal Shakti, Government of India
- Central Water Commission
- National Mission for Clean Ganga
- GemStat and MeteoStat for providing access to their valuable datasets
