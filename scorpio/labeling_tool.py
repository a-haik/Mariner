import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_folium import st_folium
import os

# Import your existing visualizers and profiler
from src.visualizer import plot_trajectory_folium, plot_series
from src.mission_profiler import MissionProfiler

# --- CONFIGURATION & DATA LOADING ---
st.set_page_config(layout="wide", page_title="Mariner Labeling Tool")

root_path = Path.cwd()
interim_dir = root_path / "data" / "interim"
labeled_dir = root_path / "data" / "labeled"
processed_dir = root_path / "data" / "processed"

# Ensure directories exist
for d in [interim_dir, labeled_dir, processed_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Sidebar File Selection
st.sidebar.title("📁 File Selection")
interim_files = [f for f in os.listdir(interim_dir) if f.endswith('.csv')]

if not interim_files:
    st.warning("No files found in data/interim/. Please run your standardization script first.")
    st.stop()

selected_file = st.sidebar.selectbox("Select Interim File to Label:", interim_files)

@st.cache_data
def load_data(filename):
    """Loads interim data, generates chunks if needed, and loads verified progress."""
    interim_path = interim_dir / filename
    blocks_path = labeled_dir / f"blocks_{filename}"
    verified_path = labeled_dir / f"verified_{filename}"
    
    # 1. Load Interim Data
    interim_df = pd.read_csv(interim_path, index_col=0, parse_dates=True)
    
    # 2. Load or Generate Blocks
    if blocks_path.exists():
        blocks_df = pd.read_csv(blocks_path, parse_dates=['Start_Time', 'End_Time'])
    else:
        with st.spinner("Generating review blocks (this only happens once)..."):
            profiler = MissionProfiler(speed_threshold=1.0)
            blocks_df = profiler.generate_review_blocks(interim_df)
            blocks_df.to_csv(blocks_path, index=False)
            
    # 3. Load Verified Progress
    if verified_path.exists():
        verified_df = pd.read_csv(verified_path, parse_dates=['Start_Time', 'End_Time'])
    else:
        verified_df = pd.DataFrame(columns=blocks_df.columns)
        
    return interim_df, blocks_df, verified_df, verified_path

interim_df, blocks_df, verified_df, verified_path = load_data(selected_file)

# --- SESSION STATE MANAGEMENT ---
if 'current_index' not in st.session_state or st.session_state.get('last_file') != selected_file:
    unverified_indices = blocks_df[~blocks_df['Block_ID'].isin(verified_df['Block_ID'])].index
    st.session_state.current_index = unverified_indices[0] if len(unverified_indices) > 0 else len(blocks_df)
    st.session_state.last_file = selected_file

# --- MAIN UI ---
st.title(f"🚢 MARINER Labeling: `{selected_file}`")

# Check if completely done
if st.session_state.current_index >= len(blocks_df):
    st.success("🎉 All blocks for this file have been verified!")
    
    if st.button("🚀 Finalize & Generate Processed Data", use_container_width=True, type="primary"):
        with st.spinner("Broadcasting labels to 5-minute telemetry..."):
            final_df = interim_df.copy()
            final_df['STATUS'] = 'unknown' # Default fallback
            
            for _, block in verified_df.iterrows():
                # Apply the verified status to the precise time window
                final_df.loc[block['Start_Time']:block['End_Time'], 'STATUS'] = block['Human_Verified_Mode']
            
            output_path = processed_dir / selected_file
            final_df.to_csv(output_path)
            st.balloons()
            st.success(f"✅ Processed dataset saved to: {output_path}")
    st.stop()

# --- LABELING VIEW ---
current_block = blocks_df.iloc[st.session_state.current_index]
block_id = current_block['Block_ID']
auto_guess = current_block['Auto_Guessed_Mode']

# Buffer for plotting context
start_time = current_block['Start_Time'] - pd.Timedelta(hours=1)
end_time = current_block['End_Time'] + pd.Timedelta(hours=1)
block_telemetry = interim_df.loc[start_time:end_time].copy()
if 'STATUS' not in block_telemetry.columns:
    block_telemetry['STATUS'] = "UNVERIFIED"

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### Block ID: `{block_id}` of {len(blocks_df)}")
    st.markdown(f"**Duration:** {current_block['Duration_h']} hours")
    st.markdown(f"**Mean Speed:** {current_block['Mean_Speed_kn']} kn")
    st.markdown(f"**Mean Power:** {current_block['Mean_Power_kW']} kW")
    
    status_options = [
        "port_idle", "port_loading", "port_unloading", 
        "sea_loitering", "sea_transit_ballast", "sea_transit_laden", 
        "unknown"
    ]
    
    default_index = status_options.index(auto_guess) if auto_guess in status_options else len(status_options)-1
    selected_status = st.selectbox("Verify Operational Mode:", options=status_options, index=default_index)
    
    if st.button("💾 Save & Next Block", use_container_width=True):
        verified_row = current_block.copy()
        verified_row['Human_Verified_Mode'] = selected_status
        verified_df.loc[len(verified_df)] = verified_row
        verified_df.to_csv(verified_path, index=False)
        st.session_state.current_index += 1
        st.rerun()

with col2:
    st.markdown("### Telemetry Profile")
    fig, axes = plot_series(block_telemetry, ['SPEED(knots)', 'AE_POWER(kW)'], subplots=True)
    st.pyplot(fig)

st.markdown("### Trajectory Map")
try:
    map_fig = plot_trajectory_folium(block_telemetry)
    st_folium(map_fig, width=1200, height=500)
except Exception as e:
    st.warning(f"Could not render map. Ensure coordinates exist. Error: {e}")