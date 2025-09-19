import streamlit as st
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import requests
import io
import re
from typing import List, Tuple, Dict, Any
import tempfile
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="KMZ/KML AGM Distance Calculator",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ KMZ/KML AGM Distance Calculator")
st.markdown("Upload KMZ/KML files to calculate distances between AGM points along centerlines")

# Google Distance Matrix API key (hardcoded as requested)
GOOGLE_API_KEY = "AIzaSyCd7sfheaJIbB8_J9Q9cxWb5jnv4U0K0LA"

def extract_coordinates_from_kml(kml_content: str) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float]]]:
    """
    Extract AGM points and centerline coordinates from KML content
    """
    try:
        root = ET.fromstring(kml_content)
        
        # Handle namespaces
        namespaces = {
            'kml': 'http://www.opengis.net/kml/2.2',
            '': 'http://www.opengis.net/kml/2.2'
        }
        
        agm_points = []
        centerline_points = []
        
        # Find all folders
        for folder in root.findall('.//kml:Folder', namespaces) + root.findall('.//Folder', namespaces):
            folder_name_elem = folder.find('.//kml:name', namespaces) or folder.find('.//name', namespaces)
            if folder_name_elem is None:
                continue
                
            folder_name = folder_name_elem.text
            
            if not folder_name:
                continue
                
            # Process AGMs folder
            if 'AGM' in folder_name.upper():
                for placemark in folder.findall('.//kml:Placemark', namespaces) + folder.findall('.//Placemark', namespaces):
                    name_elem = placemark.find('.//kml:name', namespaces) or placemark.find('.//name', namespaces)
                    coords_elem = placemark.find('.//kml:coordinates', namespaces) or placemark.find('.//coordinates', namespaces)
                    
                    if name_elem is not None and coords_elem is not None:
                        name = name_elem.text
                        coords_text = coords_elem.text.strip()
                        
                        if coords_text:
                            coords = coords_text.split(',')
                            if len(coords) >= 2:
                                try:
                                    lon, lat = float(coords[0]), float(coords[1])
                                    agm_points.append((lat, lon, name))
                                except ValueError:
                                    continue
            
            # Process CENTERLINE folder
            elif 'CENTERLINE' in folder_name.upper():
                for placemark in folder.findall('.//kml:Placemark', namespaces) + folder.findall('.//Placemark', namespaces):
                    # Look for LineString coordinates
                    linestring = placemark.find('.//kml:LineString', namespaces) or placemark.find('.//LineString', namespaces)
                    if linestring is not None:
                        coords_elem = linestring.find('.//kml:coordinates', namespaces) or linestring.find('.//coordinates', namespaces)
                        if coords_elem is not None:
                            coords_text = coords_elem.text.strip()
                            if coords_text:
                                coord_pairs = coords_text.split()
                                for coord_pair in coord_pairs:
                                    coords = coord_pair.split(',')
                                    if len(coords) >= 2:
                                        try:
                                            lon, lat = float(coords[0]), float(coords[1])
                                            centerline_points.append((lat, lon))
                                        except ValueError:
                                            continue
        
        return agm_points, centerline_points
        
    except ET.ParseError as e:
        st.error(f"Error parsing KML: {e}")
        return [], []

def extract_from_kmz(kmz_file) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float]]]:
    """
    Extract coordinates from KMZ file
    """
    agm_points = []
    centerline_points = []
    
    try:
        with zipfile.ZipFile(kmz_file, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.kml'):
                    with zip_ref.open(file_name) as kml_file:
                        kml_content = kml_file.read().decode('utf-8')
                        agm_pts, centerline_pts = extract_coordinates_from_kml(kml_content)
                        agm_points.extend(agm_pts)
                        centerline_points.extend(centerline_pts)
    except Exception as e:
        st.error(f"Error reading KMZ file: {e}")
    
    return agm_points, centerline_points

def sort_agm_points_along_centerline(agm_points: List[Tuple[float, float, str]], 
                                   centerline_points: List[Tuple[float, float]]) -> List[Tuple[float, float, str]]:
    """
    Sort AGM points based on their order along the centerline
    """
    if not centerline_points or not agm_points:
        return agm_points
    
    def distance_to_line_segment(point: Tuple[float, float], line_start: Tuple[float, float], 
                                line_end: Tuple[float, float]) -> float:
        """Calculate the distance from a point to a line segment"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line start to line end
        dx, dy = x2 - x1, y2 - y1
        
        if dx == 0 and dy == 0:
            # Line segment is actually a point
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Parameter t for the closest point on the line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        
        # Closest point on the line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance from point to closest point on line segment
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def find_position_along_centerline(agm_point: Tuple[float, float]) -> float:
        """Find the position of an AGM point along the centerline"""
        min_distance = float('inf')
        best_position = 0
        
        for i in range(len(centerline_points) - 1):
            distance = distance_to_line_segment(agm_point, centerline_points[i], centerline_points[i + 1])
            if distance < min_distance:
                min_distance = distance
                best_position = i
        
        return best_position
    
    # Create list with positions
    agm_with_positions = []
    for lat, lon, name in agm_points:
        position = find_position_along_centerline((lat, lon))
        agm_with_positions.append((position, lat, lon, name))
    
    # Sort by position along centerline
    agm_with_positions.sort(key=lambda x: x[0])
    
    # Return sorted AGM points
    return [(lat, lon, name) for _, lat, lon, name in agm_with_positions]

def get_terrain_distance(origin: Tuple[float, float], destination: Tuple[float, float]) -> float:
    """
    Get terrain-aware distance using Google Distance Matrix API
    Returns distance in meters
    """
    try:
        origin_str = f"{origin[0]},{origin[1]}"
        destination_str = f"{destination[0]},{destination[1]}"
        
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            'origins': origin_str,
            'destinations': destination_str,
            'mode': 'walking',  # This considers terrain
            'units': 'metric',
            'key': GOOGLE_API_KEY
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if (data['status'] == 'OK' and 
            data['rows'][0]['elements'][0]['status'] == 'OK'):
            distance_meters = data['rows'][0]['elements'][0]['distance']['value']
            return distance_meters
        else:
            # Fallback to haversine distance
            return haversine_distance(origin, destination) * 1000  # Convert km to meters
            
    except Exception as e:
        st.warning(f"API error, using fallback distance calculation: {e}")
        return haversine_distance(origin, destination) * 1000  # Convert km to meters

def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the great circle distance between two points on Earth (in kilometers)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r

def meters_to_feet(meters: float) -> float:
    """Convert meters to feet"""
    return meters * 3.28084

def meters_to_miles(meters: float) -> float:
    """Convert meters to miles"""
    return meters * 0.000621371

def extract_agm_number(agm_name: str) -> int:
    """Extract AGM number from name for proper sorting"""
    # Look for numbers in the AGM name
    numbers = re.findall(r'd+', agm_name)
    if numbers:
        return int(numbers[0])
    return 0

def create_distance_csv(agm_points: List[Tuple[float, float, str]]) -> pd.DataFrame:
    """
    Create CSV with AGM pairs, distances, and cumulative distances
    """
    if len(agm_points) < 2:
        return pd.DataFrame()
    
    # Sort AGM points by extracted number
    sorted_agms = sorted(agm_points, key=lambda x: extract_agm_number(x[2]))
    
    data = []
    cumulative_distance_meters = 0
    
    for i in range(len(sorted_agms) - 1):
        current_agm = sorted_agms[i]
        next_agm = sorted_agms[i + 1]
        
        # Get terrain-aware distance
        distance_meters = get_terrain_distance(
            (current_agm[0], current_agm[1]), 
            (next_agm[0], next_agm[1])
        )
        
        distance_feet = meters_to_feet(distance_meters)
        distance_miles = meters_to_miles(distance_meters)
        
        cumulative_distance_meters += distance_meters
        cumulative_distance_feet = meters_to_feet(cumulative_distance_meters)
        cumulative_distance_miles = meters_to_miles(cumulative_distance_meters)
        
        # Format AGM pair
        current_num = extract_agm_number(current_agm[2])
        next_num = extract_agm_number(next_agm[2])
        agm_pair = f"{current_num:03d} to {next_num:03d}"
        
        data.append({
            'AGM Pair': agm_pair,
            'Distance (feet)': round(distance_feet, 2),
            'Distance (miles)': round(distance_miles, 4),
            'Cumulative Distance from Start (feet)': round(cumulative_distance_feet, 2),
            'Cumulative Distance from Start (miles)': round(cumulative_distance_miles, 4)
        })
    
    return pd.DataFrame(data)

# Main application interface
uploaded_files = st.file_uploader(
    "Upload KMZ/KML files",
    type=['kmz', 'kml'],
    accept_multiple_files=True,
    help="Upload one or more KMZ or KML files containing AGM points and centerlines"
)

if uploaded_files:
    all_agm_points = []
    all_centerline_points = []
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        st.write(f"Processing file: {uploaded_file.name}")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.kmz' if uploaded_file.name.endswith('.kmz') else '.kml') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        try:
            if uploaded_file.name.endswith('.kmz'):
                agm_points, centerline_points = extract_from_kmz(tmp_file_path)
            else:  # .kml file
                kml_content = uploaded_file.getvalue().decode('utf-8')
                agm_points, centerline_points = extract_coordinates_from_kml(kml_content)
            
            all_agm_points.extend(agm_points)
            all_centerline_points.extend(centerline_points)
            
            st.success(f"✅ Found {len(agm_points)} AGM points and {len(centerline_points)} centerline points in {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    if all_agm_points and all_centerline_points:
        st.write(f"**Total: {len(all_agm_points)} AGM points and {len(all_centerline_points)} centerline points**")
        
        # Display AGM points
        if st.checkbox("Show AGM Points Details"):
            agm_df = pd.DataFrame(all_agm_points, columns=['Latitude', 'Longitude', 'Name'])
            st.dataframe(agm_df)
        
        # Sort AGM points along centerline
        with st.spinner("Sorting AGM points along centerline..."):
            sorted_agm_points = sort_agm_points_along_centerline(all_agm_points, all_centerline_points)
        
        st.success("✅ AGM points sorted along centerline")
        
        # Calculate distances
        if st.button("Calculate Distances", type="primary"):
            with st.spinner("Calculating terrain-aware distances using Google Distance Matrix API..."):
                distance_df = create_distance_csv(sorted_agm_points)
            
            if not distance_df.empty:
                st.success("✅ Distance calculations complete!")
                
                # Display the results
                st.subheader("📊 Distance Results")
                st.dataframe(distance_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_distance_miles = distance_df['Distance (miles)'].sum()
                    st.metric("Total Distance", f"{total_distance_miles:.2f} miles")
                
                with col2:
                    avg_distance_feet = distance_df['Distance (feet)'].mean()
                    st.metric("Average Segment", f"{avg_distance_feet:.0f} feet")
                
                with col3:
                    num_segments = len(distance_df)
                    st.metric("Number of Segments", num_segments)
                
                # Prepare CSV for download
                csv_buffer = io.StringIO()
                distance_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"agm_distances_{timestamp}.csv"
                
                # Download button
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    type="primary"
                )
                
                # Additional information
                st.info("💡 **Note:** Distances are calculated using Google's Distance Matrix API which considers terrain and walking paths for more accurate measurements.")
                
            else:
                st.error("❌ Could not generate distance calculations. Please check your AGM points.")
    
    elif all_agm_points:
        st.warning("⚠️ AGM points found but no centerline detected. Please ensure your file contains a 'CENTERLINE' folder with line data.")
    elif all_centerline_points:
        st.warning("⚠️ Centerline found but no AGM points detected. Please ensure your file contains an 'AGMs' folder with point data.")
    else:
        st.error("❌ No AGM points or centerline data found. Please check your file structure.")

else:
    st.info("👆 Please upload KMZ or KML files to get started.")
    
    # Instructions
    st.markdown("""
    ### 📋 Instructions:
    
    1. **Upload Files**: Select one or more KMZ or KML files containing your AGM points and centerlines
    2. **File Structure**: Ensure your files have:
       - An 'AGMs' or 'AGM' folder containing point markers
       - A 'CENTERLINE' folder containing line data
    3. **Processing**: The app will automatically extract and sort AGM points along the centerline
    4. **Calculate**: Click the "Calculate Distances" button to compute terrain-aware distances
    5. **Download**: Export your results as a CSV file
    
    ### 🎯 Features:
    - Supports both KMZ and KML file formats
    - Ignores 'MAP NOTES' and 'ACCESS' folders as requested
    - Uses Google Distance Matrix API for terrain-aware calculations
    - Sorts AGM points along the centerline path
    - Generates comprehensive distance reports with cumulative totals
    - Provides CSV export functionality
    """)
