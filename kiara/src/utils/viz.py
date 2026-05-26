import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json

from config.vessel_specs import POWER_CONFIG, DELAY_PARAMS, CLIMATE_STATS
from orchestrator import KiaraOrchestrator

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
    
    # --- COLUMN 1: SIMULATION SETUP ---
    sim_days_w = widgets.IntSlider(value=1, min=1, max=3, description='Horizon (Days):', style=style, layout=w_layout)
    start_hour_w = widgets.FloatSlider(value=0.0, min=0.0, max=24.0, step=5/60, description='View Start (Sim Hour):', style=style, layout=w_layout)
    duration_w = widgets.FloatSlider(value=4.5, min=5/60, max=24.0, step=5/60, description='View Duration (Hours):', style=style, layout=w_layout)
    trend_toggle = widgets.Checkbox(value=False, description='Overlay 1-Min Trend', style=style, layout=w_layout)
    delay_toggle = widgets.Checkbox(value=False, description='Simulate Delays', style=style, layout=w_layout)
    
    # --- COLUMN 2: WEATHER PROFILE ---
    month_dd = widgets.Dropdown(options=month_options, value='1', description='Month:', style=style, layout=w_layout)
    w0_slider = widgets.FloatSlider(value=0.15, min=0.0, max=1.0, step=0.05, description='Initial Weather Level (W0):', style=style, layout=w_layout)
    rng_seed_w = widgets.IntSlider(value=42, min=1, max=100, description='RNG Seed:', style=style, layout=w_layout)
    gust_frac = widgets.FloatSlider(value=40.0, min=0.0, max=100.0, step=5.0, description='Wind Gust Turbulence (%):', style=style, layout=w_layout)
    
    # --- COLUMN 3: MODEL PARAMETERS ---
    wave_res = widgets.FloatSlider(value=15.0, min=0.0, max=40.0, step=1.0, description='Added Wave Resistance (%):', style=style, layout=w_layout)
    sigma_frac = widgets.FloatSlider(value=3.0, min=0.0, max=10.0, step=0.5, description='Load Drift Amplitude (%):', style=style, layout=w_layout)
    delta_inst_w = widgets.FloatSlider(value=0.5, min=0.0, max=3.0, step=0.1, description='Telemetry Sensor Error (%):', style=style, layout=w_layout)
    maneuver_time_w = widgets.FloatSlider(value=6.0, min=2.0, max=12.0, step=0.5, description='Maneuver Time (Mins):', style=style, layout=w_layout)
    
    run_btn = widgets.Button(description='▶ Update Telemetry Profile', button_style='success', layout=widgets.Layout(width='340px', height='45px'))
    out_area = widgets.Output()

    def update_simulation_bounds(*args):
        max_total_hours = float(sim_days_w.value * 24)
        start_hour_w.max = max_total_hours - 0.25
        remaining_hours = max_total_hours - start_hour_w.value
        duration_w.max = max_total_hours
        if duration_w.value > remaining_hours:
            duration_w.value = max(0.25, remaining_hours)

    sim_days_w.observe(update_simulation_bounds, 'value')
    start_hour_w.observe(update_simulation_bounds, 'value')

    def update_twin(b):
        with out_area:
            clear_output(wait=True)
            np.random.seed(rng_seed_w.value)
            
            cfg = POWER_CONFIG.copy()
            
            tunable_params = {
                'wave_resistance_factor': wave_res.value * 0.01,
                'sigma_fraction': sigma_frac.value * 0.01,
                'gust_amp_fraction': gust_frac.value * 0.01,
                'delta_instrument': delta_inst_w.value * 0.01,
                'tau_human': POWER_CONFIG['tau_human']
            }
            
            # Synchronize core dispatch configuration parameters
            start_date = datetime(2026, int(month_dd.value), 3, 8, 50)
            m_key = month_dd.value if month_dd.value in climate_db else "8"
            
            orch = KiaraOrchestrator(
                schedule=schedule, start_dt=start_date, delay_params=DELAY_PARAMS, power_config=cfg,
                weather_params=climate_db[m_key]["jacobi"],
                k_factors=climate_db[m_key]["k_factors"],
                initial_weather=w0_slider.value
            )
            
            # Execute calculation arrays while routing maneuvering durations to dispatcher
            orch.dispatcher.delay_params['maneuvering']['avg_mins'] = maneuver_time_w.value
            
            # Intercept core execution path to inject dynamic maneuver parameter
            padded_seconds = (sim_days_w.value + 1) * 24 * 3600
            weather_global = orch._generate_jacobi_weather(padded_seconds)
            
            df_timeline = orch.dispatcher.generate_timeline(
                weather_array=weather_global, 
                days=sim_days_w.value, 
                enable_delays=delay_toggle.value,
                avg_maneuver_mins=maneuver_time_w.value
            )
            
            df_micro = orch.power_gen.generate_traces(
                df_timeline=df_timeline, 
                weather_global=weather_global, 
                k_factors=orch.k_factors, 
                tunable_params=tunable_params,
                month_mu=orch.weather_params['mu'],
                dt_seconds=1
            )
            
            view_start = pd.Timestamp(start_date) + pd.Timedelta(hours=start_hour_w.value)
            view_end = view_start + pd.Timedelta(hours=duration_w.value)
            
            df_slice = df_micro[(df_micro['timestamp'] >= view_start) & (df_micro['timestamp'] <= view_end)]
            
            if df_slice.empty:
                print("Error: Sliced index window contains zero frames. Verify your simulation hour coordinates.")
                return

            state_to_main = {'transit': POWER_CONFIG['P_main_sea'], 'maneuvering': POWER_CONFIG['P_main_maneuver'], 'port_dwell': POWER_CONFIG['P_main_port_base'], 'overnight_dwell': 0.0}
            state_to_aux = {'transit': POWER_CONFIG['P_aux_sea'], 'maneuvering': POWER_CONFIG['P_aux_maneuver'], 'port_dwell': POWER_CONFIG['P_aux_port_base'], 'overnight_dwell': POWER_CONFIG['P_aux_hotel']}
            nominal_main = df_slice['state'].map(state_to_main).fillna(0.0).values
            nominal_aux = df_slice['state'].map(state_to_aux).fillna(0.0).values

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [1.2, 2.5, 2]})
            signal_alpha = 0.3 if trend_toggle.value else 1.0
            
            # Tier 1: Environmental Metrics (Formatted Labels)
            ax1.plot(df_slice['timestamp'], df_slice['W_effective'], color='purple', linewidth=1.8, label="Weather Index")
            ax1.axhline(df_slice['W_cut_in'].iloc[0], color='darkorange', linestyle='--', alpha=0.8, label="Thruster Activation")
            ax1.axhline(df_slice['W_saturation'].iloc[0], color='darkred', linestyle='--', alpha=0.8, label="Thruster Saturation")
            ax1.axhline(df_slice['W_baseline'].iloc[0], color='blue', linestyle=':', alpha=0.6, label="Baseline")
            ax1.set_ylabel('Weather Index', fontweight='bold')
            ax1.set_ylim(0.0, min(1.02, df_slice['W_effective'].max() + 0.05))
            ax1.grid(True, alpha=0.15)
            ax1.legend(loc='upper right')
            ax1.set_title("KIARA Telemetry Dashboard", fontsize=14, fontweight='bold')

            # Tier 2: Propulsion System (Formatted Labels)
            ax2.plot(df_slice['timestamp'], df_slice['P_main_kW'], color='#d62728', alpha=signal_alpha, label='Signal')
            if trend_toggle.value:
                ax2.plot(df_slice['timestamp'], df_slice['P_main_kW'].rolling(60, center=True).mean(), color='darkred', linewidth=2, label='Smoothed Signal')
            ax2.plot(df_slice['timestamp'], nominal_main, color='black', linestyle=':', linewidth=1.75, drawstyle='steps-post', label='Baseline')
            ax2.set_ylabel('Main Power [kW]', fontweight='bold')
            ax2.set_ylim(0.0, df_slice['P_main_kW'].max() + 1000.0)

            # Tier 3: Electrical Infrastructure (Formatted Labels)
            ax3.plot(df_slice['timestamp'], df_slice['P_aux_kW'], color='#2ca02c', alpha=signal_alpha, linewidth=1.2, label='Signal')
            if trend_toggle.value:
                ax3.plot(df_slice['timestamp'], df_slice['P_aux_kW'].rolling(60, center=True).mean(), color='darkgreen', linewidth=2, label='Smoothed Signal')
            ax3.plot(df_slice['timestamp'], nominal_aux, color='black', linestyle=':', linewidth=1.75, drawstyle='steps-post', label='Baseline')
            ax3.set_ylabel('Auxiliary Power [kW]', fontweight='bold')
            ax3.set_xlabel('Timeline Horizon', fontweight='bold')
            ax3.set_ylim(0.0, df_slice['P_aux_kW'].max() + 40.0)
            
            if duration_w.value <= 2.0:
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            elif duration_w.value <= 24.0:
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            else:
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('Day %d, %H:%M'))

            c_map = {'transit': '#1f77b4', 'maneuvering': '#9467bd', 'port_dwell': '#ff7f0e'}
            for _, r in df_timeline.iterrows():
                if r['start_time'] > view_end or r['end_time'] < view_start: continue
                t_start = max(r['start_time'], view_start)
                t_end = min(r['end_time'], view_end)
                
                ax2.axvspan(t_start, t_end, color=c_map.get(r['state'], 'white'), alpha=0.05)
                ax3.axvspan(t_start, t_end, color=c_map.get(r['state'], 'white'), alpha=0.05)

                if delay_toggle.value and r.get('delay_mins', 0) > 0:
                    nominal_end_time = r['end_time'] - pd.Timedelta(minutes=r['delay_mins'])
                    if nominal_end_time < view_end and r['end_time'] > view_start:
                        h_start = max(nominal_end_time, view_start)
                        h_end = min(r['end_time'], view_end)
                        if h_end > h_start:
                            ax2.axvspan(h_start, h_end, hatch='//', edgecolor='red', facecolor='none', alpha=0.20)
                            ax3.axvspan(h_start, h_end, hatch='//', edgecolor='red', facecolor='none', alpha=0.20)

            for ax in [ax2, ax3]:
                ax.grid(True, alpha=0.15)
                h, l = ax.get_legend_handles_labels()
                h.extend([
                    mpatches.Patch(color='#1f77b4', alpha=0.15, label='Transit'),
                    mpatches.Patch(color='#9467bd', alpha=0.15, label='Maneuvering'),
                    mpatches.Patch(color='#ff7f0e', alpha=0.15, label='Loading')
                ])
                ax.legend(handles=h, loc='upper right')

            plt.tight_layout()
            plt.show()

    run_btn.on_click(update_twin)
    
    # Render three distinct column segments using the widgets layout engine
    panel = widgets.VBox([
        widgets.HTML("<h2> Kiara Digital Twin Calibration </h2>"),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML("<h3><b>1. Simulation Setup</b></h3>"), 
                sim_days_w, start_hour_w, duration_w, trend_toggle, delay_toggle
            ], layout=widgets.Layout(padding='0px 15px 0px 0px')),
            widgets.VBox([
                widgets.HTML("<h3><b>2. Weather Profile</b></h3>"), 
                month_dd, rng_seed_w, w0_slider, gust_frac
            ], layout=widgets.Layout(padding='0px 15px 0px 0px')),
            widgets.VBox([
                widgets.HTML("<h3><b>3. Model Parameters</b></h3>"), 
                wave_res, sigma_frac, delta_inst_w, maneuver_time_w
            ])
        ]),
        widgets.HTML("<br>"),
        run_btn,
        out_area
    ])
    display(panel)