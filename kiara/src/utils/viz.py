import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json
import os

from config.vessel_specs import POWER_CONFIG, DELAY_PARAMS, CLIMATE_STATS
from orchestrator import KiaraOrchestrator

def generate_core_plot(df_slice, df_timeline, nominal_main, nominal_aux, tunable_params, 
                       export_params, trend_toggle, delay_toggle):
    """
    Pure plotting core shared between interactive display and high-res disk export.
    Allows dynamic layout manipulation based on user formatting overrides with adaptive y-limits.
    """
    f_size = export_params['font_size']
    plt.rcParams.update({'font.size': f_size, 'axes.labelsize': f_size + 2, 'axes.titlesize': f_size + 4})
    
    included_plots = []
    if export_params['show_weather']: included_plots.append('weather')
    if export_params['show_propulsion']: included_plots.append('propulsion')
    if export_params['show_auxiliary']: included_plots.append('auxiliary')
    
    num_plots = len(included_plots)
    if num_plots == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Select at least one plot pane", ha='center', va='center')
        return fig
        
    height_ratios = []
    if export_params['show_weather']: height_ratios.append(1.2)
    if export_params['show_propulsion']: height_ratios.append(2.5)
    if export_params['show_auxiliary']: height_ratios.append(2.0)
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(export_params['width'], export_params['height']), 
                             sharex=True if num_plots > 1 else False,
                             gridspec_kw={'height_ratios': height_ratios} if num_plots > 1 else None)
    
    if num_plots == 1:
        axes = [axes]
        
    ax_idx = 0
    signal_alpha = 0.3 if trend_toggle else 1.0
    view_end = df_slice['timestamp'].max()
    view_start = df_slice['timestamp'].min()
    
    aux_axis_idx = None

    # Tier 1: Environmental Metrics Configuration
    if export_params['show_weather']:
        ax = axes[ax_idx]
        ax.plot(df_slice['timestamp'], df_slice['W_effective'], color='purple', linewidth=1.8, label="Weather Index")
        
        # Capture reference values for threshold tracking arrays
        w_cut = df_slice['W_cut_in'].iloc[0]
        w_sat = df_slice['W_saturation'].iloc[0]
        w_base = df_slice['W_baseline'].iloc[0]
        
        ax.axhline(w_cut, color='darkorange', linestyle='--', alpha=0.8, label="Thruster Activation")
        ax.axhline(w_sat, color='darkred', linestyle='--', alpha=0.8, label="Thruster Saturation")
        ax.axhline(w_base, color='blue', linestyle=':', alpha=0.6, label="Baseline")
        ax.set_ylabel('Weather Index', fontweight='bold')
        
        # Dynamic weather y-bounds: must encompass all threshold lines and active weather variations
        w_min_val = min(df_slice['W_effective'].min(), w_base, w_cut)
        w_max_val = max(df_slice['W_effective'].max(), w_sat)
        ax.set_ylim(max(0.0, w_min_val - 0.05), min(1.02, w_max_val + 0.05))
        
        ax.grid(True, alpha=0.15)
        ax.legend(loc='lower right', fontsize=f_size - 1)
        if ax_idx == 0:
            ax.set_title("KIARA High-Speed Ferry Telemetry Simulation", fontweight='bold')
        ax_idx += 1

    # Tier 2: Main Propulsion Configuration
    if export_params['show_propulsion']:
        ax = axes[ax_idx]
        ax.plot(df_slice['timestamp'], df_slice['P_main_kW'], color='#d62728', alpha=signal_alpha, label='Signal')
        if trend_toggle:
            ax.plot(df_slice['timestamp'], df_slice['P_main_kW'].rolling(300, center=True).mean(), color='darkred', linewidth=2, label='Smoothed Signal')
        ax.plot(df_slice['timestamp'], nominal_main, color='black', linestyle=':', linewidth=1.75, drawstyle='steps-post', label='Baseline')
        ax.set_ylabel('Main Power [kW]', fontweight='bold')
        
        # Adaptive Y-Limits based on local telemetry extrema
        p_main_min = min(df_slice['P_main_kW'].min(), nominal_main.min())
        p_main_max = max(df_slice['P_main_kW'].max(), nominal_main.max())
        padding_main = (p_main_max - p_main_min) * 0.05 if (p_main_max - p_main_min) > 0 else 100.0
        ax.set_ylim(max(0.0, p_main_min - padding_main), p_main_max + padding_main)
        
        ax.grid(True, alpha=0.15)
        ax.legend(loc='lower right', fontsize=f_size - 1)
        if ax_idx == 0:
            ax.set_title("KIARA High-Speed Ferry Telemetry Simulation", fontweight='bold')
        ax_idx += 1

    # Tier 3: Electrical Auxiliary Infrastructure Configuration
    if export_params['show_auxiliary']:
        ax = axes[ax_idx]
        aux_axis_idx = ax_idx  
        ax.plot(df_slice['timestamp'], df_slice['P_aux_kW'], color='#2ca02c', alpha=signal_alpha, linewidth=1.2, label='Signal')
        if trend_toggle:
            ax.plot(df_slice['timestamp'], df_slice['P_aux_kW'].rolling(300, center=True).mean(), color='darkgreen', linewidth=2, label='Smoothed Signal')
        ax.plot(df_slice['timestamp'], nominal_aux, color='black', linestyle=':', linewidth=1.75, drawstyle='steps-post', label='Baseline')
        ax.set_ylabel('Auxiliary Power [kW]', fontweight='bold')
        
        # Adaptive Y-Limits based on local telemetry extrema
        p_aux_min = min(df_slice['P_aux_kW'].min(), nominal_aux.min())
        p_aux_max = max(df_slice['P_aux_kW'].max(), nominal_aux.max())
        padding_aux = (p_aux_max - p_aux_min) * 0.05 if (p_aux_max - p_aux_min) > 0 else 20.0
        ax.set_ylim(max(0.0, p_aux_min - padding_aux), p_aux_max + padding_aux)
        
        ax.grid(True, alpha=0.15)
        ax.legend(loc='lower right', fontsize=f_size - 1)
        if ax_idx == 0:
            ax.set_title("KIARA High-Speed Ferry Telemetry Simulation", fontweight='bold')
        ax_idx += 1

    bottom_ax = axes[-1]
    bottom_ax.set_xlabel('Timeline Horizon', fontweight='bold')
    duration_hours = (view_end - view_start).total_seconds() / 3600.0
    if duration_hours <= 2.0:
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    elif duration_hours <= 24.0:
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('Day %d, %H:%M'))

    c_map = {
        'transit': '#1f77b4', 
        'maneuvering': '#9467bd', 
        'port_operations': '#ff7f0e',
        'idling': "#b6b6b6" 
    }
    for idx, ax in enumerate(axes):
        for _, r in df_timeline.iterrows():
            if r['start_time'] > view_end or r['end_time'] < view_start: continue
            t_start = max(r['start_time'], view_start)
            t_end = min(r['end_time'], view_end)
            ax.axvspan(t_start, t_end, color=c_map.get(r['state'], 'white'), alpha=0.15)

            if delay_toggle and r.get('delay_mins', 0) > 0:
                nominal_end_time = r['end_time'] - pd.Timedelta(minutes=r['delay_mins'])
                if nominal_end_time < view_end and r['end_time'] > view_start:
                    h_start = max(nominal_end_time, view_start)
                    h_end = min(r['end_time'], view_end)
                    if h_end > h_start:
                        ax.axvspan(h_start, h_end, hatch='//', edgecolor='red', facecolor='none', alpha=0.20)

        if aux_axis_idx is not None and idx == aux_axis_idx:
            h, l = ax.get_legend_handles_labels()
            h.extend([
                mpatches.Patch(color=c_map['transit'], alpha=0.15, label='Transit'),
                mpatches.Patch(color=c_map['maneuvering'], alpha=0.15, label='Maneuvering'),
                mpatches.Patch(color=c_map['port_operations'], alpha=0.15, label='Port Operations'),
                mpatches.Patch(color=c_map['idling'], alpha=0.5, label='Idling')
            ])
            ax.legend(handles=h, loc='lower right', fontsize=f_size - 1)

    plt.tight_layout()
    return fig


def build_interactive_dashboard(schedule_path, weather_path):
    with open(schedule_path, 'r') as f:
        schedule = json.load(f)
    with open(weather_path, 'r') as f:
        climate_db = json.load(f)

    style = {'description_width': '230px'}
    w_layout = widgets.Layout(width='380px')
    
    month_options = [
        ('January', '1'), ('February', '2'), ('March', '3'), ('April', '4'),
        ('May', '5'), ('June', '6'), ('July', '7'), ('August', '8'),
        ('September', '9'), ('October', '10'), ('November', '11'), ('December', '12')
    ]
    
    # --- Widgets Setup ---
    sim_days_w = widgets.IntSlider(value=1, min=1, max=3, description='Horizon (Days):', style=style, layout=w_layout)
    start_hour_w = widgets.FloatSlider(value=0.0, min=0.0, max=24.0, step=5/60, description='View Start (Sim Hour):', style=style, layout=w_layout)
    duration_w = widgets.FloatSlider(value=4.5, min=5/60, max=24.0, step=5/60, description='View Duration (Hours):', style=style, layout=w_layout)
    trend_toggle = widgets.Checkbox(value=False, description='Overlay 1-Min Trend', style=style, layout=w_layout)
    delay_toggle = widgets.Checkbox(value=False, description='Simulate Delays', style=style, layout=w_layout)
    
    month_dd = widgets.Dropdown(options=month_options, value='1', description='Month:', style=style, layout=w_layout)
    w0_slider = widgets.FloatSlider(value=0.10, min=0.0, max=1.0, step=0.01, description='Initial Weather Level (W0):', style=style, layout=w_layout)
    rng_seed_w = widgets.IntSlider(value=16, min=1, max=100, description='RNG Seed:', style=style, layout=w_layout)
    gust_frac = widgets.FloatSlider(value=20.0, min=0.0, max=100.0, step=5.0, description='Wind Gust Turbulence (%):', style=style, layout=w_layout)
    
    wave_res = widgets.FloatSlider(value=50.0, min=0.0, max=100.0, step=5.0, description='Added Wave Resistance (%):', style=style, layout=w_layout)
    sigma_frac = widgets.FloatSlider(value=5.0, min=0.0, max=20.0, step=1.0, description='Load Drift Amplitude (%):', style=style, layout=w_layout)
    delta_inst_w = widgets.FloatSlider(value=0.5, min=0.0, max=5.0, step=0.1, description='Telemetry Sensor Error (%):', style=style, layout=w_layout)
    maneuver_time_w = widgets.FloatSlider(value=5.0, min=2.0, max=12.0, step=0.5, description='Maneuver Time (Mins):', style=style, layout=w_layout)
    
    font_size_w = widgets.IntSlider(value=13, min=8, max=24, description='Presentation Font Size:', style=style, layout=w_layout)
    fig_width_w = widgets.FloatSlider(value=16.0, min=8.0, max=24.0, step=0.5, description='Figure Width:', style=style, layout=w_layout)
    fig_height_w = widgets.FloatSlider(value=11.0, min=6.0, max=20.0, step=0.5, description='Figure Height:', style=style, layout=w_layout)
    
    show_weather_w = widgets.Checkbox(value=True, description='Include Weather Pane', style=style, layout=w_layout)
    show_prop_w = widgets.Checkbox(value=True, description='Include Propulsion Pane', style=style, layout=w_layout)
    show_aux_w = widgets.Checkbox(value=True, description='Include Auxiliary Pane', style=style, layout=w_layout)
    
    export_filename_w = widgets.Text(value="kiara_validation_trace.png", description="Filename:", style=style, layout=w_layout)
    export_btn = widgets.Button(description='📷 Save High-Res Snapshot', button_style='info', icon='camera', layout=widgets.Layout(width='340px', height='45px'))
    
    run_btn = widgets.Button(description='▶ Update Telemetry Profile', button_style='success', layout=widgets.Layout(width='340px', height='45px'))
    out_area = widgets.Output()

    cached_data = {"df_slice": None, "df_timeline": None, "nominal_main": None, "nominal_aux": None}

    state_to_main = {
        'transit': POWER_CONFIG['P_main_transit'], 
        'maneuvering': POWER_CONFIG['P_main_maneuver_base'], 
        'port_operations': POWER_CONFIG['P_main_port_ops'], 
        'idling': POWER_CONFIG['P_main_idling']
    }
    state_to_aux = {
        'transit': POWER_CONFIG['P_aux_transit'], 
        'maneuvering': POWER_CONFIG['P_aux_maneuver_base'], 
        'port_operations': POWER_CONFIG['P_aux_port_ops'], 
        'idling': POWER_CONFIG['P_aux_idling']
    }

    def update_simulation_bounds(*args):
        max_total_hours = float(sim_days_w.value * 24)
        start_hour_w.max = max_total_hours - 0.25
        remaining_hours = max_total_hours - start_hour_w.value
        duration_w.max = max_total_hours
        if duration_w.value > remaining_hours:
            duration_w.value = max(0.25, remaining_hours)

    sim_days_w.observe(update_simulation_bounds, 'value')
    start_hour_w.observe(update_simulation_bounds, 'value')

    def run_pipeline():
        """Uses the unified orchestrator pipeline to run the full simulation."""
        np.random.seed(rng_seed_w.value)
        cfg = POWER_CONFIG.copy()
        
        tunable_params = {
            'wave_resistance_factor': wave_res.value * 0.01,
            'sigma_fraction': sigma_frac.value * 0.01,
            'gust_amp_fraction': gust_frac.value * 0.01, 
            'delta_instrument': delta_inst_w.value * 0.01,
            'tau_human': POWER_CONFIG['tau_human']
        }
        
        start_date = datetime(2026, int(month_dd.value), 3, 8, 50)
        m_key = month_dd.value if month_dd.value in climate_db else "8"
        
        orch = KiaraOrchestrator(
            schedule=schedule, start_dt=start_date, delay_params=DELAY_PARAMS, power_config=cfg,
            weather_params=climate_db[m_key]["jacobi"], k_factors=climate_db[m_key]["k_factors"],
            initial_weather=w0_slider.value
        )
        
        orch.dispatcher.delay_params['maneuvering']['avg_mins'] = maneuver_time_w.value
        
        df_timeline, df_micro = orch.run_simulation(
            tunable_params=tunable_params,
            days=sim_days_w.value,
            enable_delays=delay_toggle.value,
            departure_hour=9
        )
        
        view_start = pd.Timestamp(start_date) + pd.Timedelta(hours=start_hour_w.value)
        view_end = view_start + pd.Timedelta(hours=duration_w.value)
        df_slice = df_micro[(df_micro['timestamp'] >= view_start) & (df_micro['timestamp'] <= view_end)].copy()
        
        return df_slice, df_timeline, tunable_params

    def update_twin(b):
        with out_area:
            clear_output(wait=True)
            df_slice, df_timeline, tunable_params = run_pipeline()
            
            if df_slice.empty:
                print("Error: Sliced index window contains zero frames. Verify your simulation hour coordinates.")
                return
            
            nominal_main = df_slice['state'].map(state_to_main).fillna(0.0).values
            nominal_aux = df_slice['state'].map(state_to_aux).fillna(0.0).values

            cached_data["df_slice"] = df_slice
            cached_data["df_timeline"] = df_timeline
            cached_data["nominal_main"] = nominal_main
            cached_data["nominal_aux"] = nominal_aux

            export_params = {
                'font_size': font_size_w.value, 'width': fig_width_w.value, 'height': fig_height_w.value,
                'show_weather': show_weather_w.value, 'show_propulsion': show_prop_w.value, 'show_auxiliary': show_aux_w.value
            }

            fig = generate_core_plot(df_slice, df_timeline, nominal_main, nominal_aux, tunable_params,
                                     export_params, trend_toggle.value, delay_toggle.value)
            plt.show()

    def export_snapshot(b):
        with out_area:
            if cached_data["df_slice"] is None:
                df_slice, df_timeline, _ = run_pipeline()
                cached_data["df_slice"] = df_slice
                cached_data["df_timeline"] = df_timeline
                cached_data["nominal_main"] = df_slice['state'].map(state_to_main).fillna(0.0).values
                cached_data["nominal_aux"] = df_slice['state'].map(state_to_aux).fillna(0.0).values

            export_params = {
                'font_size': font_size_w.value, 'width': fig_width_w.value, 'height': fig_height_w.value,
                'show_weather': show_weather_w.value, 'show_propulsion': show_prop_w.value, 'show_auxiliary': show_aux_w.value
            }
            
            fig = generate_core_plot(cached_data["df_slice"], cached_data["df_timeline"], 
                                     cached_data["nominal_main"], cached_data["nominal_aux"], {}, 
                                     export_params, trend_toggle.value, delay_toggle.value)
            
            out_dir = "../docs"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            target_path = os.path.join(out_dir, export_filename_w.value)
            fig.savefig(target_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f" Success! High-resolution presentation asset exported cleanly to: {target_path}")

    run_btn.on_click(update_twin)
    export_btn.on_click(export_snapshot)
    
    snapshot_panel = widgets.VBox([
        widgets.HTML("<h4><b>Fine-Tune Visual Layout Parameters for Slides</b></h4>"),
        widgets.HBox([
            widgets.VBox([font_size_w, fig_width_w, fig_height_w]),
            widgets.VBox([show_weather_w, show_prop_w, show_aux_w])
        ]),
        widgets.HTML("<br>"),
        widgets.HBox([export_filename_w, export_btn], layout=widgets.Layout(align_items='center'))
    ])
    
    accordion_w = widgets.Accordion(children=[snapshot_panel], selected_index=None)
    accordion_w.set_title(0, " Presentation Export Controls")
    
    panel = widgets.VBox([
        widgets.HTML("<h2> MARINER Calibration Workspace: KIARA Fleet Twin</h2>"),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML("<h3><b>1. Simulation Setup</b></h3>"), 
                sim_days_w, start_hour_w, duration_w, trend_toggle, delay_toggle
            ], layout=widgets.Layout(padding='0px 15px 0px 0px')),
            widgets.VBox([
                widgets.HTML("<h3><b>2. Weather Profile</b></h3>"), 
                month_dd, w0_slider, rng_seed_w, gust_frac
            ], layout=widgets.Layout(padding='0px 15px 0px 0px')),
            widgets.VBox([
                widgets.HTML("<h3><b>3. Model Parameters</b></h3>"), 
                wave_res, sigma_frac, delta_inst_w, maneuver_time_w
            ])
        ]),
        widgets.HTML("<br>"),
        accordion_w,
        widgets.HTML("<br>"),
        run_btn,
        out_area
    ])
    display(panel)