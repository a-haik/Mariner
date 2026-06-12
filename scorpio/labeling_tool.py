import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_folium import st_folium
import os
import plotly.express as px # Added for global map

from src.visualizer import plot_trajectory_folium, plot_series
from src.mission_profiler import MissionProfiler

st.set_page_config(layout="wide", page_title="Mariner Labeling Tool")

root_path = Path.cwd()
interim_dir = root_path / "data" / "interim"
labeled_dir = root_path / "data" / "labeled"
processed_dir = root_path / "data" / "processed"

for d in [interim_dir, labeled_dir, processed_dir]:
    d.mkdir(parents=True, exist_ok=True)

st.sidebar.title("📁 File Selection")
interim_files = [f for f in os.listdir(interim_dir) if f.endswith('.csv')]

if not interim_files:
    st.warning("No files found in data/interim/.")
    st.stop()

selected_file = st.sidebar.selectbox("Select Interim File to Label:", interim_files)

@st.cache_data
def load_interim_data(filename):
    interim_path = interim_dir / filename
    return pd.read_csv(interim_path, index_col=0, parse_dates=True)

interim_df = load_interim_data(selected_file)
blocks_path = labeled_dir / f"blocks_{selected_file}"

def init_blocks():
    if blocks_path.exists():
        df = pd.read_csv(blocks_path, parse_dates=['Start_Time', 'End_Time'])
        # Force the column to be text so Pandas doesn't assume it's float64
        df['Human_Verified_Mode'] = df['Human_Verified_Mode'].astype(object).fillna("")
        return df
    else:
        with st.spinner("Generating review blocks..."):
            profiler = MissionProfiler(speed_threshold=1.0)
            df = profiler.generate_review_blocks(interim_df)
            # Force the column to be text
            df['Human_Verified_Mode'] = df['Human_Verified_Mode'].astype(object).fillna("")
            df.to_csv(blocks_path, index=False)
            return df

if 'blocks_df' not in st.session_state or st.session_state.get('last_file') != selected_file:
    st.session_state.blocks_df = init_blocks()
    st.session_state.last_file = selected_file
    
    unverified = st.session_state.blocks_df[st.session_state.blocks_df['Human_Verified_Mode'].isna() | (st.session_state.blocks_df['Human_Verified_Mode'] == "")]
    st.session_state.current_index = unverified.index[0] if len(unverified) > 0 else 0

def recalculate_block_stats(block_row):
    mask = (interim_df.index >= block_row['Start_Time']) & (interim_df.index <= block_row['End_Time'])
    chunk = interim_df.loc[mask]
    if not chunk.empty:
        speed_col = next((c for c in chunk.columns if 'SPEED' in c.upper()), 'SPEED(knots)')
        block_row['Duration_h'] = round((chunk.index.max() - chunk.index.min()).total_seconds() / 3600.0, 2)
        block_row['Mean_Speed_kn'] = round(chunk[speed_col].mean(), 2)
        block_row['Mean_Power_kW'] = round(chunk['AE_POWER(kW)'].mean(), 1)
    return block_row

# --- MAIN UI ---
st.title(f"🚢 MARINER Labeling: `{selected_file}`")

blocks_df = st.session_state.blocks_df
curr_idx = st.session_state.current_index

if curr_idx >= len(blocks_df):
    st.success("🎉 All blocks verified! You can step back to edit, or finalize.")
    curr_idx = len(blocks_df) - 1 
    
current_block = blocks_df.iloc[curr_idx]
block_id = current_block['Block_ID']
auto_guess = current_block['Auto_Guessed_Mode']

start_time = current_block['Start_Time'] - pd.Timedelta(hours=1)
end_time = current_block['End_Time'] + pd.Timedelta(hours=1)
block_telemetry = interim_df.loc[start_time:end_time].copy()
if 'STATUS' not in block_telemetry.columns:
    block_telemetry['STATUS'] = "UNVERIFIED"

# --- TOP NAVIGATION & FINALIZATION ---
nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 2])
with nav_col1:
    if st.button("⬅️ Previous Block", use_container_width=True, disabled=(curr_idx == 0)):
        st.session_state.current_index -= 1
        st.rerun()
with nav_col2:
    if st.button("Next Block ➡️", use_container_width=True, disabled=(curr_idx >= len(blocks_df) - 1)):
        st.session_state.current_index += 1
        st.rerun()
with nav_col3:
    if st.button("🚀 Finalize & Generate Processed Data", use_container_width=True, type="primary"):
        with st.spinner("Broadcasting labels and generating global plot (this may take a few seconds)..."):
            final_df = interim_df.copy()
            
            # Start with empty values instead of 'unknown'
            final_df['STATUS'] = pd.NA 
            
            for _, block in blocks_df.iterrows():
                final_df.loc[block['Start_Time']:block['End_Time'], 'STATUS'] = block['Human_Verified_Mode']
            
            # Auto-patch dropped edge ticks
            final_df['STATUS'] = final_df['STATUS'].bfill().ffill()
            
            final_df.to_csv(processed_dir / selected_file)
            
        st.success(f"✅ Dataset saved to: data/processed/{selected_file}")
        st.balloons()
        
        # --- NEW VISUALIZATION FEATURE ---
        st.markdown("### 🏆 Final Verified Telemetry Profile")
        st.info("Plotting your entire voyage dataset with verified operational modes...")
        
        # Call your existing plot_series function on the completed dataset
        fig, axes = plot_series(final_df, ['SPEED(knots)', 'AE_POWER(kW)'], subplots=True, status_col='STATUS')
        st.pyplot(fig)
        
        st.stop()

# --- ADVANCED BLOCK EDITOR ---
with st.expander("🛠️ Advanced Block Editing (Split & Merge)"):
    edit_col1, edit_col2 = st.columns(2)
    
    with edit_col1:
        internal_mask = (interim_df.index > current_block['Start_Time']) & (interim_df.index < current_block['End_Time'])
        valid_split_times = interim_df[internal_mask].index
        
        if len(valid_split_times) > 0:
            split_time = st.selectbox("Select Split Timestamp:", valid_split_times)
            if st.button("✂️ Split Block Here"):
                new_block = current_block.copy()
                current_block['End_Time'] = split_time
                new_block['Start_Time'] = split_time + pd.Timedelta(minutes=5)
                
                blocks_df.iloc[curr_idx] = recalculate_block_stats(current_block)
                new_block = recalculate_block_stats(new_block)
                
                blocks_df = pd.concat([blocks_df.iloc[:curr_idx+1], pd.DataFrame([new_block]), blocks_df.iloc[curr_idx+1:]]).reset_index(drop=True)
                blocks_df['Block_ID'] = range(1, len(blocks_df) + 1)
                
                st.session_state.blocks_df = blocks_df
                blocks_df.to_csv(blocks_path, index=False)
                st.rerun()

    with edit_col2:
        st.write("Merge with Neighbors:")
        if curr_idx > 0 and st.button("⬆️ Merge with Previous"):
            prev_block = blocks_df.iloc[curr_idx - 1]
            prev_block['End_Time'] = current_block['End_Time']
            blocks_df.iloc[curr_idx - 1] = recalculate_block_stats(prev_block)
            
            blocks_df = blocks_df.drop(curr_idx).reset_index(drop=True)
            blocks_df['Block_ID'] = range(1, len(blocks_df) + 1)
            
            st.session_state.current_index -= 1
            st.session_state.blocks_df = blocks_df
            blocks_df.to_csv(blocks_path, index=False)
            st.rerun()
            
        if curr_idx < len(blocks_df) - 1 and st.button("⬇️ Merge with Next"):
            next_block = blocks_df.iloc[curr_idx + 1]
            current_block['End_Time'] = next_block['End_Time']
            blocks_df.iloc[curr_idx] = recalculate_block_stats(current_block)
            
            blocks_df = blocks_df.drop(curr_idx + 1).reset_index(drop=True)
            blocks_df['Block_ID'] = range(1, len(blocks_df) + 1)
            
            st.session_state.blocks_df = blocks_df
            blocks_df.to_csv(blocks_path, index=False)
            st.rerun()

st.divider()

# --- LABELING VIEW ---
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
    
    current_val = current_block.get('Human_Verified_Mode', auto_guess)
    default_index = status_options.index(current_val) if current_val in status_options else len(status_options)-1
    
    selected_status = st.selectbox("Verify Operational Mode:", options=status_options, index=default_index)
    
    if st.button("💾 Save Label", use_container_width=True):
        # Use .loc and force selected_status to be a string
        blocks_df.loc[curr_idx, 'Human_Verified_Mode'] = str(selected_status)
        blocks_df.to_csv(blocks_path, index=False)
        st.session_state.blocks_df = blocks_df
        st.success("Saved!")

with col2:
    st.markdown("### Telemetry Profile")
    fig, axes = plot_series(block_telemetry, ['SPEED(knots)', 'AE_POWER(kW)'], subplots=True)
    
    for ax in axes:
        ax.axvline(x=current_block['Start_Time'], color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=current_block['End_Time'], color='red', linestyle='--', alpha=0.7)
    st.pyplot(fig)

# --- TRAJECTORY MAPS (TWO TABS) ---
st.markdown("### Trajectory Context")

# Universal Toggle for both maps
plot_points_toggle = st.checkbox("📍 Plot trajectories as discrete points (Check this to fix 'teleporting' GPS lines)", value=True)

tab1, tab2 = st.tabs(["📍 Current Block Trajectory", "🌍 Global Voyage Context"])

with tab1:
    try:
        # Pass the universal toggle state to your Folium generator
        map_fig = plot_trajectory_folium(block_telemetry, plot_as_points=plot_points_toggle)
        st_folium(map_fig, width=1200, height=500)
    except Exception as e:
        st.warning(f"Could not render local map. Error: {e}")

with tab2:
    try:
        # Downsample the 5-month dataset to every 30 mins to keep the app lightning fast
        global_df = interim_df[['LATITUDE(DD)', 'LONGITUDE(DD)']].dropna().copy()
        global_df = global_df.iloc[::6] 
        global_df['Context'] = "Full Voyage"
        
        # Highlight the current block in bright red
        mask = (global_df.index >= current_block['Start_Time']) & (global_df.index <= current_block['End_Time'])
        global_df.loc[mask, 'Context'] = "Current Selection"
        
        # Apply the universal toggle logic to Plotly
        if plot_points_toggle:
            # Renders as discrete scatter points
            fig_global = px.scatter_mapbox(
                global_df, lat="LATITUDE(DD)", lon="LONGITUDE(DD)", 
                color="Context",
                color_discrete_map={"Full Voyage": "rgba(128, 128, 128, 0.4)", "Current Selection": "red"},
                zoom=1, mapbox_style="carto-positron",
                height=500
            )
            # Make the red active dot larger
            fig_global.update_traces(marker=dict(size=8), selector=dict(name="Current Selection"))
        else:
            # Renders as continuous connected lines
            fig_global = px.line_mapbox(
                global_df, lat="LATITUDE(DD)", lon="LONGITUDE(DD)", 
                color="Context",
                color_discrete_map={"Full Voyage": "rgba(128, 128, 128, 0.4)", "Current Selection": "red"},
                zoom=1, mapbox_style="carto-positron",
                height=500
            )
            # Make the red active line thicker
            fig_global.update_traces(line=dict(width=4), selector=dict(name="Current Selection"))
            
        fig_global.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        st.plotly_chart(fig_global, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render global map. Error: {e}")