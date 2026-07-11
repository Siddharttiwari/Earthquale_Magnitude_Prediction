import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Pre-declare model variable to satisfy VS Code linter and clear red lines
rf = None

# Load the best performing model (Random Forest) safely
try:
    rf = joblib.load("random_forest.pkl")
except FileNotFoundError:
    st.error("Model file missing: random_forest.pkl. Make sure it is in your project directory.")

# Streamlit Page Configuration
st.set_page_config(page_title="Live Earthquake Magnitude Predictor", layout="wide")
st.title("Earthquake Magnitude Prediction & Risk Assessment Dashboard")
st.markdown(
    "Enter a city or country name below. The system will pull **live real-time seismic data** "
    "from the USGS global database, map the active hazard zones, and evaluate infrastructure safety risks."
)

# --- Helper Function: Fetch Real-Time Data from USGS API ---
def get_live_usgs_data(lat, lon, days_back):
    """
    Queries the USGS API for recent earthquake events within a 250km radius 
    of the target location for a custom number of days back.
    Returns average depth, average station count, total event count, and a DataFrame of coordinates.
    """
    start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&latitude={lat}&longitude={lon}&maxradiuskm=250&starttime={start_date}"
    
    try:
        response = requests.get(url, timeout=5).json()
        features = response.get('features', [])
        
        map_data = []
        depths = []
        stations = []
        
        if features:
            for f in features:
                coords = f['geometry']['coordinates']
                lon_val = coords[0]
                lat_val = coords[1]
                depth_val = coords[2]
                nst_val = f['properties']['nst']
                
                # Collect all coordinates for our map layer
                map_data.append({'latitude': lat_val, 'longitude': lon_val})
                
                if depth_val is not None:
                    depths.append(depth_val)
                if nst_val is not None:
                    stations.append(nst_val)
            
            avg_depth = sum(depths) / len(depths) if depths else 10.0
            avg_nst = int(sum(stations) / len(stations)) if stations else 20
            
            return avg_depth, avg_nst, len(features), pd.DataFrame(map_data)
        else:
            return 10.0, 20, 0, pd.DataFrame(columns=['latitude', 'longitude'])
            
    except Exception:
        return 10.0, 20, 0, pd.DataFrame(columns=['latitude', 'longitude'])

# --- 1. Search Configurations ---
st.markdown("### Search Configurations")
col_input, col_slider = st.columns([2, 1])

with col_input:
    location_input = st.text_input("Enter City, Region, or Country Name", placeholder="e.g., Tokyo, Japan")

with col_slider:
    days_window = st.slider("Select Historical Data Window (Days)", min_value=1, max_value=90, value=30, step=1)

latitude = None
longitude = None

if location_input:
    geolocator = Nominatim(user_agent="earthquake_predictor_app")
    try:
        location = geolocator.geocode(location_input)
        if location:
            latitude = location.latitude
            longitude = location.longitude
            st.success(f"📍 Location Found: {location.address}")
        else:
            st.error("Location not found. Please check the spelling or be more specific.")
    except GeocoderTimedOut:
        st.error("Connection timed out. Please try searching again.")

# --- 2. Live Feature Extraction & Analysis Pipeline ---
st.markdown("---")

if latitude is not None and longitude is not None:
    if st.button("Fetch Live Data & Analyze Seismic Risk", type="primary"):
        if rf is not None:
            
            with st.spinner(f"Streaming live tectonic profiles from the last {days_window} days via USGS API..."):
                live_depth, live_nst, total_events, recent_quakes_df = get_live_usgs_data(latitude, longitude, days_window)
            
            # Layout columns for data metrics and maps side-by-side
            col_metrics, col_map = st.columns([1, 1])
            
            with col_metrics:
                st.markdown(f"### 📊 Live Regional Seismic Profile")
                st.metric("Recent Nearby Events", f"{total_events} quakes")
                st.metric("Calculated Average Depth", f"{live_depth:.2f} km")
                st.metric("Avg Monitoring Station Linkage", f"{live_nst} stations")
                
                # Predict via Random Forest
                features_df = pd.DataFrame(
                    [[latitude, longitude, live_depth, live_nst]], 
                    columns=['Latitude(deg)', 'Longitude(deg)', 'Depth(km)', 'No_of_Stations']
                )
                pred_mag = rf.predict(features_df)[0]
                
                st.markdown("---")
                st.success(f"### ⚡ Predicted Potential Magnitude: {pred_mag:.2f}")
                
                # --- 3. Vulnerability & Safety Radius Assessment Logic ---
                st.markdown("### 🛡️ Safety & Impact Radius Assessment")
                
                # Seismological distance threshold estimation based on predicted strength
                if pred_mag < 4.0:
                    safety_radius = 15  # km
                    st.success("✅ Low Threat Level")
                    st.info(f"**Impact:** Minor shaking. Shaking might be felt up to **{safety_radius} km** from the hypocenter. Negligible threat to modern structural masonry.")
                elif 4.0 <= pred_mag < 5.5:
                    safety_radius = 50  # km
                    st.warning("⚠️ Moderate Threat Level")
                    st.warning(f"**Impact:** Moderate seismic energy. Vibrations noticeably felt within a **{safety_radius} km** radius. Slight damage possible to old, non-reinforced buildings.")
                else:
                    safety_radius = 150  # km
                    st.error("🚨 High Threat Level")
                    st.error(f"**Impact:** Severe structural risk. Destructive shockwaves could impact infrastructure within a **{safety_radius} km** zone. High caution advised for structural stability.")
            
            with col_map:
                st.markdown("### 🗺️ Geospatial Hazard Mapping")
                if not recent_quakes_df.empty:
                    st.caption("🔴 Red dots show exact epicenters of tracked seismic activities around your target area.")
                    # Interactive map automatically scales and focuses directly over your coordinates
                    st.map(recent_quakes_df, color="#ff4b4b", size=25)
                else:
                    # If no historical events, just map the single searched location point
                    target_point = pd.DataFrame([{'latitude': latitude, 'longitude': longitude}])
                    st.caption("🔵 Blue dot represents the searched city center (No recent regional tremors tracked).")
                    st.map(target_point, color="#4b86ff", size=40)
                    
        else:
            st.error("The Random Forest engine file (.pkl) could not be loaded successfully.")
else:
    st.warning("Please enter a valid city or country name above to fetch coordinates before predicting.")