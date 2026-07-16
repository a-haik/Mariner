# python/src/utils/plotting.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

    ax.set_ylabel('Cumulative Cost [$]')
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
        avg_cost = df.loc['Average', 'Total Cost [$]']
    else:
        df_runs = df
        avg_cost = None

    fig, ax = plt.subplots(figsize=(10, 5))
    
    costs = df_runs['Total Cost [$]']
    x_labels = df_runs.index

    if plot_type == 'bar':
        ax.bar(x_labels, costs, color='royalblue', alpha=0.8, edgecolor='black')
        ax.set_ylabel("Total Cost [$]")
        plt.xticks(rotation=45, ha='right')
        
    elif plot_type == 'line':
        ax.plot(x_labels, costs, marker='o', color='darkorange', linewidth=2, markersize=8)
        ax.set_ylabel("Total Cost [$]")
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    if avg_cost is not None:
        ax.axhline(avg_cost, color='red', linestyle='--', linewidth=1.5, label=f'Average Cost: {avg_cost:.2f} $')
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

def plot_markov_matrix(mc_model: dict, title: str = "Markov Transition Matrix", ax=None, save_plot: bool = False):
    """
    Renders the transition matrix as a heatmap.
    Can be plotted as a standalone figure or embedded into an existing axes (ax).
    """
    # If no axis is provided, create a standalone figure
    is_standalone = ax is None
    if is_standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    P = mc_model['P']
    levels = mc_model['levels']
    
    # Generate the heatmap
    cax = ax.imshow(P, cmap='viridis', aspect='auto', origin='lower')
    
    # Format the physical kW levels on the axes
    ax.set_xticks(np.arange(len(levels)))
    ax.set_yticks(np.arange(len(levels)))
    
    labels = [f"{int(lvl)}" for lvl in levels]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    ax.set_xlabel("Next State Demand [kW]")
    ax.set_ylabel("Current State Demand [kW]")
    ax.set_title(title, fontweight='bold', fontsize=12)
    
    # Add colorbar (ax.figure ensures it attaches to the correct parent figure)
    cbar = ax.figure.colorbar(cax, ax=ax)
    cbar.set_label("Transition Probability", rotation=270, labelpad=15)
    
    # Handle standalone rendering and saving
    if is_standalone:
        plt.tight_layout()
        if save_plot:
            os.makedirs('figures', exist_ok=True)
            safe_title = title.replace(" ", "_").lower()
            plt.savefig(f'figures/{safe_title}.png', dpi=300)
        plt.show()

def plot_simulation_dashboard(df: pd.DataFrame, config, terminal_costs=(0.0, 0.0), mc_model=None, title: str = "Simulation Dashboard", indiv: bool = False, save_plot: bool = False):
    """
    Dynamically generates a comprehensive dashboard of a single simulation run.
    Now supports Phase 3 terminal boundary condition visualization.
    """
    t_hours = df['time'] / 3600.0  
    p_module = np.where(df['n_active'] > 0, df['p_fc_actual'] / df['n_active'], 0.0)
    
    t_hours_jump = t_hours.tolist()
    cum_cost_o = df['cost_o'].cumsum().tolist()
    cum_cost_s = df['cost_s'].cumsum().tolist()
    cum_cost_total = df['cost_total'].cumsum().tolist()
    
    # Check for battery
    has_battery = 'p_batt_actual' in df.columns and 'soc' in df.columns
    if has_battery:
        cum_cost_bat = df['cost_bat'].cumsum().tolist()

    # Check for transient costs
    has_transient = 'cost_tr' in df.columns
    if has_transient:
        cum_cost_tr = df['cost_tr'].cumsum().tolist()

    # --- APPLY TERMINAL JUMP (t = T) ---
    term_n_cost, term_soc_cost = terminal_costs
    if term_n_cost != 0.0 or term_soc_cost != 0.0:
        # Duplicate the final time coordinate to create a sharp vertical line
        t_hours_jump.append(t_hours_jump[-1])
        
        cum_cost_o.append(cum_cost_o[-1]) # Operating cost doesn't jump
        cum_cost_s.append(cum_cost_s[-1] + term_n_cost)
        
        if has_battery:
            cum_cost_bat.append(cum_cost_bat[-1] + term_soc_cost)
            
        if has_transient:
            cum_cost_tr.append(cum_cost_tr[-1]) # Transient cost doesn't jump
            
        cum_cost_total.append(cum_cost_total[-1] + term_n_cost + term_soc_cost)

    # 2. Setup Figure Layout
    fig = None
    ax_pwr, ax_mod, ax_cost, ax_bat, ax_mc = None, None, None, None, None
    
    if not indiv:
        if mc_model is None:
            # Standard 2x2 Grid Layout
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            ax_pwr = axes[0, 0]
            ax_mod = axes[0, 1]
            ax_cost = axes[1, 0]
            
            ax_bat = axes[1, 1]
            if not has_battery:
                ax_bat.set_visible(False) # Hide if testing FC-Only Plant
                
        else:
            # Asymmetric 5-Plot GridSpec Layout
            fig = plt.figure(figsize=(16, 14))
            gs = GridSpec(3, 2, figure=fig)
            fig.suptitle(f"{title} (with Markov Analytics)", fontsize=16, fontweight='bold')
            
            # Row 0 spans both columns [0, :]
            ax_mod = fig.add_subplot(gs[0, :])   
            
            # Rows 1 & 2 populate the bottom 2x2 space
            ax_pwr = fig.add_subplot(gs[1, 0])
            ax_cost = fig.add_subplot(gs[1, 1])
            if has_battery:
                ax_bat = fig.add_subplot(gs[2, 0])
            ax_mc = fig.add_subplot(gs[2, 1])

    def get_ax(target_ax):
        """Yields a new figure if indiv=True, otherwise yields the GridSpec target."""
        if indiv:
            temp_fig, ax = plt.subplots(figsize=(10, 4))
            return temp_fig, ax
        return fig, target_ax

    # =========================================================
    # --- PANE 1: Power Split ---
    # =========================================================
    current_fig, ax1 = get_ax(ax_pwr)
    ax1.plot(t_hours, df['P_d'], label='Raw Demand (P_d)', color='black', alpha=0.3, linewidth=2)
    ax1.plot(t_hours, df['p_fc_actual'], label='Fuel Cell Power', color='royalblue', linewidth=1.5)
    
    ax1.set_ylabel("Power [kW]")
    if indiv:
        ax1.set_title(f"{title} - Power Split")
    else:
         ax1.set_title("Demand vs Fuel Cell Production", fontweight='bold')
    
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    if indiv:
        plt.show()

    # =========================================================
    # --- PANE 2: Module Kinematics ---
    # =========================================================
    current_fig, ax2 = get_ax(ax_mod)
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
        ax2.set_title(f"{title} - Kinematics")
    else:
        ax2.set_title("Module Activation & Power Distribution", fontweight='bold')
    
    ax2.grid(True, linestyle='--', alpha=0.6)
    if indiv:
        plt.show()

    # =========================================================
    # --- PANE 3: Cumulative Economics ---
    # =========================================================
    current_fig, ax3 = get_ax(ax_cost)
    
    # Use t_hours_jump and our new appended lists
    ax3.plot(t_hours_jump, cum_cost_total, label='Total Cost', color='black', linewidth=2.5)
    ax3.plot(t_hours_jump, cum_cost_o, label='Operating Cost (FC)', color='royalblue', linestyle='--')
    ax3.plot(t_hours_jump, cum_cost_s, label='Switching Cost', color='crimson', linestyle='--')
    
    if has_battery:
        ax3.plot(t_hours_jump, cum_cost_bat, label='Battery Degradation', color='darkorange', linestyle='--')

    # Add the transient cost plot
    if has_transient:
        ax3.plot(t_hours_jump, cum_cost_tr, label='Transient Cost', color='mediumorchid', linestyle='--')
        
    ax3.set_ylabel("Cumulative Cost [$]")
    ax3.set_yscale('symlog', linthresh=1.0)
    
    if indiv:
        ax3.set_title(f"{title} - Economics")
    else:
        ax3.set_title("Cumulative Costs Evolution", fontweight='bold')
    
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.6)
    if indiv:
        plt.show()

    # =========================================================
    # --- PANE 4: Battery Health & Power (Hybrid Only) ---
    # =========================================================
    if has_battery and (ax_bat is not None or indiv):
        current_fig, ax4 = get_ax(ax_bat)

        # Left Axis: State of Charge (SoC)
        ax4.plot(t_hours, df['soc'] * 100, label='State of Charge', color='black', linewidth=2)
        ax4.set_ylabel("SoC [%]")
        ax4.set_ylim([0, 100])
        # 1. Dynamically fetch safety limits from config (defaults fallback to 20% / 80%)
        soc_min = getattr(config, 'soc_min', 0.2) * 100
        soc_max = getattr(config, 'soc_max', 0.8) * 100
        soc_init = getattr(config, 'soc_initial', 0.5) * 100
        
        # Plot safety boundary lines
        ax4.axhline(soc_max, color='black', linewidth=1, linestyle='--', alpha=0.7)
        ax4.axhline(soc_min, color='black', linewidth=1, linestyle='--', alpha=0.7)
        
        # 2. Handle Initial and Target SoC Guidelines dynamically
        if hasattr(config, 'soc_target'):
            soc_tgt = config.soc_target * 100
            
            # Check if initial and target are practically identical (using tolerance for floats)
            if abs(soc_tgt - soc_init) < 1e-3:
                ax4.axhline(soc_tgt, color='blue', linewidth=1.5, linestyle='-.', 
                            label=f'Initial & Target SoC ({int(soc_tgt)}%)')
            else:
                ax4.axhline(soc_init, color='gray', linewidth=1.5, linestyle=':', 
                            label=f'Initial SoC ({int(soc_init)}%)')
                ax4.axhline(soc_tgt, color='blue', linewidth=1.5, linestyle='-.', 
                            label=f'Target SoC ({int(soc_tgt)}%)')
        else:
            # Fallback if no target exists but we still want to show where it started
            ax4.axhline(soc_init, color='gray', linewidth=1.5, linestyle=':', 
                        label=f'Initial SoC ({int(soc_init)}%)')
        
# Right Axis: Battery Power Area Plot
        ax4_twin = ax4.twinx()
        ax4_twin.plot(t_hours, df['p_batt_actual'], color='gray', linewidth=0.5, alpha=0.5)

        p_max = df['p_batt_actual'].abs().max()
        limit_p = max(p_max * 1.1, 1.0) # Graceful fallback if battery is unused (max=0)
        
        # 1. Find where soc_init sits proportionally on the left axis (0 to 100 scale)
        soc_fraction = soc_init / 100.0
        
        # Clamp the fraction slightly so the power plot doesn't disappear 
        # off-screen if soc_init happens to be exactly 0% or 100%
        soc_fraction = max(0.1, min(0.9, soc_fraction)) 
        
        # 2. Calculate the total required span for the right y-axis.
        # We check both the positive and negative required sides to ensure no clipping.
        total_span = max(limit_p / (1 - soc_fraction), limit_p / soc_fraction)
        
        # 3. Set the asymmetric limits
        ax4_twin.set_ylim([-total_span * soc_fraction, total_span * (1 - soc_fraction)])
        # ----------------------------------------
        
        # Fill positive/negative
        ax4_twin.fill_between(t_hours, 0, df['p_batt_actual'], 
                         where=(df['p_batt_actual'] > 0), 
                         color='red', alpha=0.2, label='Discharging', interpolate=True)
                         
        ax4_twin.fill_between(t_hours, 0, df['p_batt_actual'], 
                         where=(df['p_batt_actual'] < 0), 
                         color='green', alpha=0.2, label='Charging', interpolate=True)
        
        ax4_twin.set_ylabel("Battery Power [kW]")
        
        if indiv:
            ax4.set_title(f"{title} - Battery State")
        else:
            ax4.set_title("Battery Utilization & State of Charge", fontweight='bold')
        
        # Legend combiner
        lines_1, labels_1 = ax4.get_legend_handles_labels()
        lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
        by_label = dict(zip(labels_1 + labels_2, lines_1 + lines_2)) 
        ax4.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        ax4.grid(True, linestyle='--', alpha=0.6)
        if indiv:
            plt.show()

    # =========================================================
    # --- PANE 5: Markov Transition Matrix ---
    # =========================================================
    if mc_model is not None:
        if indiv:
            plot_markov_matrix(mc_model, title=f"{title} - Markov Matrix")
        else:
            plot_markov_matrix(mc_model, title="Trained Markov Transition Probabilities", ax=ax_mc)

    # =========================================================
    # 3. Final Formatting (for Dashboard view)
    # =========================================================
    if not indiv:
        # Add X-labels only to the bottom-most plots to keep it clean
        if mc_model is None:
            ax_cost.set_xlabel("Time [Hours]")
            if has_battery:
                ax_bat.set_xlabel("Time [Hours]")
        else:
            if has_battery:
                ax_bat.set_xlabel("Time [Hours]")
            # ax_mc label is handled inherently by plot_markov_matrix
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        if save_plot:
            os.makedirs('figures', exist_ok=True)
            safe_title = title.replace(" ", "_").lower()
            plt.savefig(f'figures/{safe_title}_dashboard.png', dpi=300)
        
        plt.show()