import streamlit as st
import os
import folium
from streamlit_folium import folium_static


# Locations with their coordinates (latitude, longitude)
LOCATIONS = {
    "Rudraprayag": (30.2833, 79.0167),
    "Devprayag": (30.1407, 78.5936),
    "Bijnor": (29.37, 78.14),
    "Kannauj": (27.0648, 79.9120),
    "Kanpur": (26.4499, 80.3319),
    "Allahabad": (25.4358, 81.8463),
    "Murshidabad": (24.0988, 88.2679),
    "Ghazipur": (25.5799, 83.5963),
    "Patna": (25.5941, 85.1376),
    "Naini": (25.3871, 81.8597),
    "Bhagalpur": (25.271603, 87.025665),
    "Parganas": (22.1848, 88.1891)  # Combined location
}

# Configure the page
st.set_page_config(
    page_title="AquaVision AI",
    page_icon="🌊",
    layout="wide"
)

def create_ganga_river_map():
    """
    Create an interactive map of the Ganga River basin
    
    Returns:
        folium.Map: Configured map object
    """
    # Create a map centered on the middle of the Ganga basin
    m = folium.Map(
        location=[25.5, 80], 
        zoom_start=6,
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite'
    )

    # Add markers for each location
    for name, (lat, lon) in LOCATIONS.items():
        # Custom HTML popup with styled content
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; color: #333; padding: 10px;">
            <h3 style="color: #0077be; margin-bottom: 5px;">{name}</h3>
            <p style="margin: 0;">Latitude: {lat:.4f}</p>
            <p style="margin: 0;">Longitude: {lon:.4f}</p>
        </div>
        """
        
        # Add marker with popup
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=name,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # Draw river path (simplified approximation)
    river_path = list(LOCATIONS.values())
    
    # Add river path
    folium.PolyLine(
        locations=river_path, 
        color='blue', 
        weight=3, 
        opacity=0.8,
        dash_array='10'
    ).add_to(m)

    return m

# Set session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Main content rendering
def render_home_page():
    # Custom CSS with light blue gradient background and modern aesthetics
    st.markdown("""
    <style>
        .stApp { 
            background: linear-gradient(125deg, #E6F3FF 0%, #F0F8FF 100%); 
            background-size: 400% 400%; 
            animation: gradientBG 15s ease infinite; 
        }

        @keyframes gradientBG { 
            0% { background-position: 0% 50%; } 
            50% { background-position: 100% 50%; } 
            100% { background-position: 0% 50%; } 
        }

        .big-title { 
            font-family: 'Inter', sans-serif; 
            color: #0077be; 
            font-size: 5.5rem !important; 
            font-weight: 900 !important; 
            margin-bottom: 0 !important; 
            line-height: 1.1 !important; 
            text-align: center !important; 
            letter-spacing: -2px; 
            text-shadow: 0 0 30px rgba(0, 119, 190, 0.5), 0 0 60px rgba(0, 119, 190, 0.3); 
            animation: glow 3s ease-in-out infinite alternate; 
        }

        @keyframes glow { 
            0% { text-shadow: 0 0 30px rgba(0, 119, 190, 0.5); } 
            100% { text-shadow: 0 0 30px rgba(0, 119, 190, 0.8), 0 0 60px rgba(0, 119, 190, 0.6); } 
        }

        .subtitle { 
            font-family: 'Inter', sans-serif; 
            background: linear-gradient(to right, #0077be, #00b3b3); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            font-size: 1.6rem !important; 
            text-align: center !important; 
            margin: 1rem 0 3rem 0 !important; 
            font-weight: 500 !important; 
            letter-spacing: -0.5px; 
        }

        .button-card { 
            cursor: pointer; 
            border-radius: 24px; 
            padding: 2.5rem 2rem; 
            background: rgba(255, 255, 255, 0.6); 
            backdrop-filter: blur(10px); 
            border: 1px solid rgba(0, 119, 190, 0.2); 
            margin: 1rem; 
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); 
            overflow: hidden; 
            position: relative; 
        }

        .button-card:hover { 
            transform: translateY(-8px) scale(1.02); 
            border: 1px solid rgba(0, 119, 190, 0.4); 
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1), 0 0 40px rgba(0, 119, 190, 0.2); 
        }

        .emoji-icon { 
            font-size: 3rem !important; 
            margin-bottom: 1rem !important; 
            text-align: center; 
            display: block; 
            animation: bounce 2s infinite; 
        }

        @keyframes bounce { 
            0%, 100% { transform: translateY(0); } 
            50% { transform: translateY(-10px); } 
        }

        .card-title { 
            color: #0077be !important; 
            font-size: 2rem !important; 
            font-weight: 700 !important; 
            margin-bottom: 1rem !important; 
            text-align: center; 
            letter-spacing: -0.5px; 
        }

        .card-description { 
            color: rgba(0, 119, 190, 0.9) !important; 
            font-size: 1.1rem !important; 
            margin-bottom: 1.5rem !important; 
            text-align: center; 
            line-height: 1.6 !important; 
        }

        .map-container {
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 119, 190, 0.2);
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="big-title">AquaVision AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ganga Riverwater Monitoring and Analysis</p>', unsafe_allow_html=True)

    # Satellite Map Section
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    ganga_map = create_ganga_river_map()
    folium_static(ganga_map, width=1000, height=600)
    st.markdown('</div>', unsafe_allow_html=True)

    # Create two columns for the buttons
    col1, col2 = st.columns(2)

    # Feature 1: Forecast Ganga Riverwater Quality
    with col1:
        st.markdown("""
            <div class="button-card">
                <span class="emoji-icon">🌊</span>
                <h3 class="card-title">Riverwater Quality</h3>
                <p class="card-description">
                    Advanced forecasting of Ganga river water quality using AI-driven predictive analytics.
                </p>
            </div>
        """, unsafe_allow_html=True)

        if st.button('Analyze Water Quality', key='water-quality-btn'):
            st.session_state.page = 'water_quality'

    # Feature 2: Scenario Modeling
    with col2:
        st.markdown("""
            <div class="button-card">
                <span class="emoji-icon">💬</span>
                <h3 class="card-title">Give Feedback</h3>
                <p class="card-description">
                    Help us improve AquaVision AI by sharing your thoughts and suggestions.
                </p>
            </div>
        """, unsafe_allow_html=True)

        if st.button('Submit Feedback', key='feedback-btn'):
            st.session_state.page = 'feedback'

# Navigation logic
if st.session_state.page == 'home':
    render_home_page()
else:
    if st.session_state.page == 'water_quality':
        st.header("Select Location for Water Quality Analysis")
        
        # Dictionary mapping locations to their specific file paths
        location_paths = {
            "Rudraprayag": r"Location Pages\water_quality_rudraprayag.py",
            "Bhagalpur": r"Location Pages/water_quality_bhagalpur.py",
            "Devprayag": r"Location Pages/water_quality_devprayag.py",
            "Kannauj": r"Location Pages/water_quality_kannauj.py",
            "Kanpur": r"Location Pages/water_quality_kanpur.py",
            "Ghazipur": r"Location Pages/water_quality_ghazipur.py",
            "Patna": r"Location Pages/water_quality_patna.py",
            "Prayagraj": r"Ganga_Project/Location Pages/water_quality_prayagraj.py",
            "Naini": r"Location Pages/water_quality_naini.py",
            "Parganas": r"Location Pages/water_quality_parganas.py",
            "Murshidabad": r"Location Pages/water_quality_murshidabad.py",
            "Bijnor": r"Location Pages/water_quality_bijnor.py"
        }

        locations = list(location_paths.keys())
        selected_location = st.selectbox("Choose a location:", locations)
        
        if st.button('Run Analysis for Selected Location'):
            try:
                with open(location_paths[selected_location], encoding="utf-8") as file:
                    exec(file.read())
            except FileNotFoundError:
                st.error(f"Module for {selected_location} not found.")

        if st.button('Back to Home'):
            st.session_state.page = 'home'

    elif st.session_state.page == 'feedback':
        if st.button('Back to Home'):
            st.session_state.page = 'home'

        exec(open(r"feedback.py", encoding="utf-8").read())
