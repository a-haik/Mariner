import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def plot_micro_power_trace(df_micro, window_minutes=60, start_idx=0):
    """
    Plots a highly detailed window of the power trace.
    Perfect for showing the expert the transition from transit to port maneuvering.
    """
    # Slice the dataframe for the specified window
    end_idx = start_idx + (window_minutes * 60)
    df_slice = df_micro.iloc[start_idx:end_idx]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3, 2]})
    
    # 1. Meso Weather Panel
    ax1.plot(df_slice['timestamp'], df_slice['W_global'], color='purple', linewidth=1.5)
    ax1.set_ylabel('Weather Hazard\nW(t) [0-1]', fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Operational Window: Environmental & Power Profile", fontsize=14, fontweight='bold')

    # 2. Main Engine Panel (Diesel - High Inertia)
    ax2.plot(df_slice['timestamp'], df_slice['P_main_kW'], color='#d62728', label='Main Engines (Diesel)')
    ax2.fill_between(df_slice['timestamp'], df_slice['P_main_kW'], color='#d62728', alpha=0.1)
    ax2.set_ylabel('Main Power [kW]', fontweight='bold')
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    # Highlight port vs transit states using background spans
    for state, color in zip(['port_dwell', 'transit'], ['#ff7f0e', '#1f77b4']):
        state_mask = df_slice['state'] == state
        if state_mask.any():
            ax2.fill_between(df_slice['timestamp'], 0, 25000, where=state_mask, 
                             color=color, alpha=0.05, label=f'State: {state.capitalize()}' if color=='#1f77b4' else "")

    # 3. Auxiliary Engine Panel (Electric - Low Inertia / Spikes)
    ax3.plot(df_slice['timestamp'], df_slice['P_aux_kW'], color='#2ca02c', label='Auxiliary (Electric / Thrusters)')
    ax3.set_ylabel('Aux Power [kW]', fontweight='bold')
    ax3.set_xlabel('Time', fontweight='bold')
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis time nicely
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.tight_layout()
    plt.show()

def plot_ab_weather_comparison(df_calm, df_harsh, window_minutes=30):
    """
    Creates a side-by-side comparison of the auxiliary power (bow thrusters)
    during port operations in Calm vs. Harsh weather.
    Use this to calibrate the expert on the threshold parameter!
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    
    # Slice the first 30 mins of both dataframes (assuming they start in port)
    slice_calm = df_calm.head(window_minutes * 60)
    slice_harsh = df_harsh.head(window_minutes * 60)
    
    ax1.plot(slice_calm['timestamp'], slice_calm['P_aux_kW'], color='blue')
    ax1.set_title("Scenario A: Calm Weather (W ≈ 0.2)\nSmooth Baseline Auxiliary Load")
    ax1.set_ylabel("Power [kW]")
    
    ax2.plot(slice_harsh['timestamp'], slice_harsh['P_aux_kW'], color='red')
    ax2.set_title("Scenario B: Harsh Meltemi (W ≈ 0.8)\nBow Thruster Spikes Triggered")
    
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
    plt.tight_layout()
    plt.show()