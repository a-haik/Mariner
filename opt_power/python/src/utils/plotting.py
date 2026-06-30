# python/src/utils/plotting.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _get_unique_filepath(base_filepath: str) -> str:
    """
    Helper function to prevent overwriting existing plots.
    Appends _1, _2, etc. to the filename if it already exists.
    """
    if not os.path.exists(base_filepath):
        return base_filepath
    
    base_dir = os.path.dirname(base_filepath)
    filename, ext = os.path.splitext(os.path.basename(base_filepath))
    
    counter = 1
    while True:
        new_filepath = os.path.join(base_dir, f"{filename}_{counter}{ext}")
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1

def plot_dynamic_history(simulators, controller_names, title="Dynamic History Tracking", show_plot=True, save_plot=False):
    """
    Dynamically generates subplots based on the keys present in the Simulator's history dictionary.
    """
    if not simulators:
        return
        
    history_keys = [k for k in simulators[0].history.keys() if k != 'time']
    num_controllers = len(simulators)
    num_metrics = len(history_keys)
    
    fig, axes = plt.subplots(num_metrics, num_controllers, figsize=(6 * num_controllers, 2.5 * num_metrics), sharex='col')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if num_controllers == 1 and num_metrics == 1:
        axes = np.array([[axes]])
    elif num_controllers == 1:
        axes = np.expand_dims(axes, axis=1)
    elif num_metrics == 1:
        axes = np.expand_dims(axes, axis=0)
        
    time_array = simulators[0].history['time']
    
    for c_idx, sim in enumerate(simulators):
        for m_idx, key in enumerate(history_keys):
            ax = axes[m_idx, c_idx]
            data = sim.history[key]
            
            color = 'darkorange' if 'cost' in key else 'teal' if 'soc' in key else 'royalblue'
            
            ax.plot(time_array, data, label=key, color=color, linewidth=1.5)
            
            if m_idx == 0:
                ax.set_title(f"{controller_names[c_idx]}\n{key}")
            else:
                ax.set_title(key)
                
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper right')
            
            if m_idx == num_metrics - 1:
                ax.set_xlabel("Time [s]")
                
    plt.tight_layout()
    
    if save_plot:
        os.makedirs('figures', exist_ok=True)
        safe_path = _get_unique_filepath('figures/dynamic_history.png')
        plt.savefig(safe_path, dpi=300)
        
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_cost_comparison(simulators, controller_names, title='Cost Breakdown Comparison Across Strategies', show_plot=True, save_plot=False):
    """
    Generates a grouped bar chart summarizing cumulative costs.
    """
    cost_keys = [k for k in simulators[0].history.keys() if k.startswith('cost_') and k != 'cost_total']
    
    x = np.arange(len(controller_names))
    width = 0.8 / len(cost_keys) if cost_keys else 0.5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, key in enumerate(cost_keys):
        totals = [np.sum(sim.history[key]) for sim in simulators]
        offset = (i - len(cost_keys) / 2 + 0.5) * width
        
        rects = ax.bar(x + offset, totals, width, label=key)
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Cumulative Cost [€]')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(controller_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs('figures', exist_ok=True)
        safe_path = _get_unique_filepath('figures/cost_comparison.png')
        plt.savefig(safe_path, dpi=300)
        
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_benchmarker_results(df: pd.DataFrame, title: str, plot_type: str = 'bar', show_plot=True, save_plot=False):
    """
    Visualizes the outputs from the VoyageBenchmarker.
    """
    if 'Average' in df.index:
        df_runs = df.drop('Average')
        avg_cost = df.loc['Average', 'Total Cost [€]']
    else:
        df_runs = df
        avg_cost = None

    fig, ax = plt.subplots(figsize=(10, 5))
    
    costs = df_runs['Total Cost [€]']
    x_labels = df_runs.index

    if plot_type == 'bar':
        ax.bar(x_labels, costs, color='royalblue', alpha=0.8, edgecolor='black')
        ax.set_ylabel("Total Cost [€]")
        plt.xticks(rotation=45, ha='right')
        
    elif plot_type == 'line':
        ax.plot(x_labels, costs, marker='o', color='darkorange', linewidth=2, markersize=8)
        ax.set_ylabel("Total Cost [€]")
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    if avg_cost is not None:
        ax.axhline(avg_cost, color='red', linestyle='--', linewidth=1.5, label=f'Average Cost: {avg_cost:.2f} €')
        ax.legend()

    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    
    if save_plot:
        os.makedirs('figures', exist_ok=True)
        safe_title = title.replace(" ", "_").lower()
        safe_path = _get_unique_filepath(f'figures/{safe_title}.png')
        plt.savefig(safe_path, dpi=300)
        
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_simulation_dashboard(df: pd.DataFrame, title: str = "Simulation Dashboard", indiv: bool = False, save_plot: bool = False):
    """
    Dynamically generates a comprehensive dashboard (or individual plots) of a single simulation run.
    Automatically detects if the DataFrame contains hybrid battery data or just FC legacy data.
    """
    # 1. Prepare derived data
    t_hours = df['time'] / 3600.0  # Convert seconds to hours for cleaner X-axis
    
    # Safely calculate power per module (avoid divide-by-zero when n_active == 0)
    p_module = np.where(df['n_active'] > 0, df['p_fc_actual'] / df['n_active'], 0.0)
    
    # Cumulative Costs
    cum_cost_o = df['cost_o'].cumsum()
    cum_cost_s = df['cost_s'].cumsum()
    cum_cost_total = df['cost_total'].cumsum()
    
    has_battery = 'p_batt_actual' in df.columns and 'soc' in df.columns
    if has_battery:
        cum_cost_bat = df['cost_bat'].cumsum()

    num_panes = 4 if has_battery else 3

    # 2. Setup Figure Layout
    if not indiv:
        fig, axes = plt.subplots(num_panes, 1, figsize=(12, 3 * num_panes), sharex=True)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        ax_idx = 0
    
    def get_ax():
        """Helper to yield the correct axis depending on the `indiv` flag."""
        nonlocal ax_idx
        if indiv:
            # Rename to temp_fig so we don't shadow the outer 'fig' variable!
            temp_fig, ax = plt.subplots(figsize=(10, 4))
            return temp_fig, ax
        else:
            ax = axes[ax_idx]
            ax_idx += 1
            return fig, ax

    # --- PANE 1: Power Split ---
    current_fig, ax1 = get_ax()
    ax1.plot(t_hours, df['P_d'], label='Raw Demand (P_d)', color='black', alpha=0.3, linewidth=2)
    ax1.plot(t_hours, df['p_fc_actual'], label='Fuel Cell Power', color='royalblue', linewidth=1.5)
    
    # Battery power removed from here to prevent clutter
    
    ax1.set_ylabel("Power [kW]")
    if indiv:
        ax1.set_title(f"{title} - Power Split")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    if indiv:
        plt.show()

    # --- PANE 2: Module Kinematics ---
    current_fig, ax2 = get_ax()
    color_n = 'teal'
    ax2.step(t_hours, df['n_active'], label='Active Modules (n)', color=color_n, linewidth=2, where='post')
    ax2.set_ylabel("Module Count", color=color_n)
    ax2.tick_params(axis='y', labelcolor=color_n)
    
    ax2_twin = ax2.twinx()
    color_p = 'purple'
    ax2_twin.plot(t_hours, p_module, label='Power per Module', color=color_p, linewidth=1.5, alpha=0.6)
    ax2_twin.set_ylabel("Power / Module [kW]", color=color_p)
    ax2_twin.tick_params(axis='y', labelcolor=color_p)
    
    if indiv:
        ax2.set_title(f"{title} - Module Kinematics")
    ax2.grid(True, linestyle='--', alpha=0.6)
    if indiv:
        plt.show()

    # --- PANE 3: Cumulative Economics ---
    current_fig, ax3 = get_ax()
    ax3.plot(t_hours, cum_cost_total, label='Total Cost', color='black', linewidth=2.5)
    ax3.plot(t_hours, cum_cost_o, label='Operating Cost (FC)', color='royalblue', linestyle='--')
    ax3.plot(t_hours, cum_cost_s, label='Switching Cost', color='crimson', linestyle='--')
    
    if has_battery:
        ax3.plot(t_hours, cum_cost_bat, label='Battery Degradation', color='darkorange', linestyle='--')
    
    ax3.set_ylabel("Cumulative Cost [€]")
    
    # FIX: Use a symmetric log scale to handle orders of magnitude gracefully
    ax3.set_yscale('symlog', linthresh=1.0)
    
    if indiv:
        ax3.set_title(f"{title} - Economics")
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.6)
    if indiv:
        plt.show()

    # --- PANE 4: Battery Health & Power (Hybrid Only) ---
    if has_battery:
        current_fig, ax4 = get_ax()
        
        # Left Axis: Battery Power Area Plot
        ax4.plot(t_hours, df['p_batt_actual'], color='gray', linewidth=0.5, alpha=0.5) # Faint outline
        
        # Fill positive (Discharging) with Red
        ax4.fill_between(t_hours, 0, df['p_batt_actual'], 
                         where=(df['p_batt_actual'] > 0), 
                         color='red', alpha=0.2, label='Discharging', interpolate=True)
                         
        # Fill negative (Charging) with Green
        ax4.fill_between(t_hours, 0, df['p_batt_actual'], 
                         where=(df['p_batt_actual'] < 0), 
                         color='green', alpha=0.2, label='Charging', interpolate=True)
        
        ax4.set_ylabel("Battery Power [kW]")
        ax4.axhline(0, color='black', linewidth=1, linestyle='--') # Clear Zero Line
        
        # Right Axis: State of Charge (SoC)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(t_hours, df['soc'] * 100, label='State of Charge', color='black', linewidth=2)
        ax4_twin.set_ylabel("SoC [%]")
        ax4_twin.set_ylim([0, 100])
        
        if indiv:
            ax4.set_title(f"{title} - Battery State")
        
        # Combine legends from both the left and right axes cleanly
        lines_1, labels_1 = ax4.get_legend_handles_labels()
        lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
        by_label = dict(zip(labels_1 + labels_2, lines_1 + lines_2)) # Removes duplicates
        ax4.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        ax4.grid(True, linestyle='--', alpha=0.6)
        if indiv:
            plt.show()

    # 3. Final Formatting (for Dashboard view)
    if not indiv:
        axes[-1].set_xlabel("Time [Hours]")
        plt.tight_layout()
        
        if save_plot:
            os.makedirs('figures', exist_ok=True)
            safe_title = title.replace(" ", "_").lower()
            plt.savefig(f'figures/{safe_title}_dashboard.png', dpi=300)
        
        plt.show()