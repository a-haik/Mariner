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

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 9,
    'lines.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.4
})

def plot_simulation_dashboard(df: pd.DataFrame, config, terminal_costs=(0.0, 0.0), 
                             mc_model=None, title: str = "Simulation Dashboard", 
                             indiv: bool = False, save_plot: bool = False):
    """
    Publication-ready dashboard for Elsevier cas-dc (Applied Energy standard).
    - indiv=False: Full 2-column figure (width = 16.6 cm / ~6.54 in)
    - indiv=True : Single-column figure (width = 8.0 cm / ~3.15 in)
    """
    t_hours = df['time'] / 3600.0
    p_module = np.where(df['n_active'] > 0, df['p_fc_actual'] / df['n_active'], 0.0)
    
    # Flags
    has_split_fc = 'cost_h2' in df.columns and 'cost_fc_deg' in df.columns
    has_battery = 'p_batt_actual' in df.columns and 'soc' in df.columns
    has_transient = 'cost_tr' in df.columns

    # Terminal costs
    term_n_cost, term_soc_cost = terminal_costs

    # --- COST BREAKDOWN CALCULATION (FOR COST SUBPLOT) ---
    # Total accumulated costs over 24h including boundary conditions
    cost_data = {}
    if has_split_fc:
        cost_data['Fuel'] = df['cost_h2'].sum()
        cost_data['Steady Wear'] = df['cost_fc_deg'].sum()
    else:
        cost_data['FC Oper.'] = df['cost_o'].sum()

    cost_data['Switching'] = df['cost_s'].sum() + term_n_cost
    
    if has_battery:
        cost_data['Battery'] = df['cost_bat'].sum() + term_soc_cost
    if has_transient:
        cost_data['Load Change'] = df['cost_tr'].sum()

    # Dimensions (Inches: 1 cm = 0.3937 in)
    # Reduced vertical height slightly for better page economy in Elsevier papers
    if indiv:
        figsize = (3.15, 2.3)
    else:
        figsize = (6.54, 4.5)

    if not indiv:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False)
        ax_pwr = axes[0, 0]
        ax_mod = axes[0, 1]
        ax_bat = axes[1, 0]   # Switched: Battery is now (c)
        ax_cost = axes[1, 1]  # Switched: Costs are now (d)
        if not has_battery:
            ax_bat.set_visible(False)
    
    def get_ax(target_ax):
        if indiv:
            temp_fig, ax = plt.subplots(figsize=figsize)
            return temp_fig, ax
        return fig, target_ax

    time_ticks = np.arange(0, 25, 4)

    # =========================================================
    # --- PANE 1 (a): Power Split ---
    # =========================================================
    current_fig, ax1 = get_ax(ax_pwr)
    ax1.plot(t_hours, df['P_d'], label='Power Demand', color='#333333', alpha=0.35, linewidth=0.9)
    ax1.plot(t_hours, df['p_fc_actual'], label='FC Power', color='#1f77b4', linewidth=1.2)
    
    ax1.set_ylabel("Power [kW]")
    ax1.set_xlim([0, 24])
    ax1.set_xticks(time_ticks)
    ax1.grid(True, linestyle='--')
    ax1.legend(loc='upper right', frameon=True, framealpha=0.8)
    
    # Subfigure label
    ax1.text(0.02, 0.92, "(a)", transform=ax1.transAxes, fontweight='bold', fontsize=9)
    if indiv:
        ax1.set_xlabel("Time [h]")
        plt.tight_layout()
        if save_plot:
            plt.savefig(f'figures/{title}_pane_a.pdf', dpi=300)
        plt.show()

    # =========================================================
    # --- PANE 2 (b): Module Kinematics ---
    # =========================================================
    current_fig, ax2 = get_ax(ax_mod)
    color_n = '#d62728'  # Red/Teal accent
    color_p = '#1f77b4'  # Blue accent

    ax2.step(t_hours, df['n_active'], label='Active Modules Count', color=color_n, linewidth=0.9, alpha=0.7, where='post')
    ax2.set_ylabel("Active Modules $n$ [-]", color=color_n)
    ax2.tick_params(axis='y', labelcolor=color_n)
    ax2.set_xlim([0, 24])
    ax2.set_xticks(time_ticks)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t_hours, p_module, label='Individual Module Power', color=color_p, linewidth=1.2, alpha=1)
    ax2_twin.set_ylabel("Individual Module Power [kW]", color=color_p)
    ax2_twin.tick_params(axis='y', labelcolor=color_p)
    
    ax2.grid(True, linestyle='--')
    ax2.text(0.02, 0.92, "(b)", transform=ax2.transAxes, fontweight='bold', fontsize=9)
    
    if indiv:
        ax2.set_xlabel("Time [h]")
        plt.tight_layout()
        if save_plot:
            plt.savefig(f'figures/{title}_pane_b.pdf', dpi=300)
        plt.show()

    # =========================================================
    # --- PANE 3 (c): Battery Utilization & SOC (SWITCHED TO C) ---
    # =========================================================
    if has_battery and (ax_bat is not None or indiv):
        current_fig, ax4 = get_ax(ax_bat)

        # Config Limits
        soc_min = getattr(config, 'soc_min', 0.2) * 100
        soc_max = getattr(config, 'soc_max', 0.8) * 100
        soc_init = getattr(config, 'soc_initial', 0.5) * 100

        # --- RIGHT AXIS: Battery Power Fill ---
        ax4_twin = ax4.twinx()
        p_max = df['p_batt_actual'].abs().max()
        limit_p = max(p_max * 1.1, 1.0)
        
        # Align twin y-axis proportionally around soc_init
        soc_fraction = max(0.1, min(0.9, soc_init / 100.0)) 
        total_span = max(limit_p / (1 - soc_fraction), limit_p / soc_fraction)
        ax4_twin.set_ylim([-total_span * soc_fraction, total_span * (1 - soc_fraction)])

        # Single Fill for Battery Power
        ax4_twin.fill_between(
            t_hours, 0, df['p_batt_actual'], 
            color='#2ca02c', alpha=0.5, 
            edgecolor='none', linewidth=0.0,
            label='Battery Power', zorder=3, interpolate=False
        )
        ax4_twin.set_ylabel("Battery Power [kW]", color='#2ca02c')
        ax4_twin.tick_params(axis='y', labelcolor='#2ca02c')

        # --- LEFT AXIS: SOC Curve & Safety Boundaries ---
        ax4.plot(t_hours, df['soc'] * 100, label='SOC', color='black', linewidth=1.2, zorder=5)
        ax4.set_ylabel("State Of Charge [%]")
        ax4.set_ylim([0, 100])
        ax4.set_xlim([0, 24])
        ax4.set_xticks(time_ticks)
        
        # Safety Limit Lines (Dotted)
        ax4.axhline(soc_max, color='black', linewidth=0.8, linestyle=':', alpha=0.6, zorder=4)
        ax4.axhline(soc_min, color='black', linewidth=0.8, linestyle=':', alpha=0.6, zorder=4)
        
        # SOC Target / Initial Guidelines
        if hasattr(config, 'soc_target'):
            soc_tgt = config.soc_target * 100
            ax4.axhline(soc_tgt, color='k', linewidth=0.9, linestyle='--', alpha=0.75, zorder=4, label=r"SOC_0")

        # Formatting & Layer Hierarchy
        ax4.set_zorder(ax4_twin.get_zorder() + 1)
        ax4.patch.set_visible(False)
        ax4.grid(True, linestyle='--', alpha=0.5, zorder=1)
        
        # Labelled (c) now
        ax4.text(0.02, 0.92, "(c)", transform=ax4.transAxes, fontweight='bold', fontsize=9)

        # # Clean Consolidated Legend
        # lines_1, labels_1 = ax4.get_legend_handles_labels()
        # lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
        # by_label = dict(zip(labels_1 + labels_2, lines_1 + lines_2)) 
        # ax4.legend(by_label.values(), by_label.keys(), loc='upper right', 
        #            frameon=True, framealpha=0.85, edgecolor='none')

        if indiv:
            ax4.set_xlabel("Time [h]")
            plt.tight_layout()
            if save_plot:
                plt.savefig(f'figures/{title}_pane_c.pdf', dpi=300)
            plt.show()

    # =========================================================
    # --- PANE 4 (d): Cost Breakdown (SWITCHED TO D) ---
    # =========================================================
    current_fig, ax3 = get_ax(ax_cost)
    
    categories = list(cost_data.keys())
    values = list(cost_data.values())
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd'][:len(categories)]
    
    bars = ax3.bar(categories, values, color=colors, edgecolor='black', linewidth=0.6, width=0.55)
    ax3.set_ylabel("Final Cost [$]")
    
    # Scientific handling for scale differences: Symlog scale
    ax3.set_yscale('symlog', linthresh=10.0)
    ax3.set_ylim([0, max(values) * 10])
    ax3.grid(True, which='both', linestyle='--', axis='y')
    
    # Annotate values above bars
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'${height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6.5, rotation=0)

    # Labelled (d) now
    ax3.text(0.02, 0.92, "(d)", transform=ax3.transAxes, fontweight='bold', fontsize=9)
    if indiv:
        plt.tight_layout()
        if save_plot:
            plt.savefig(f'figures/{title}_pane_d.pdf', dpi=300)
        plt.show()

    # =========================================================
    # --- FINAL FORMATTING (2-COLUMN VIEW) ---
    # =========================================================
    if not indiv:
        # Bottom X-Labels
        ax_bat.set_xlabel("Time [h]")
        ax_cost.set_xlabel("Cost Category")
        ax_pwr.set_xlabel("Time [h]")
        ax_cost.tick_params(axis='x', rotation=15)
        
        plt.tight_layout(pad=0.4)
        
        if save_plot:
            os.makedirs('figures', exist_ok=True)
            safe_title = title.replace(" ", "_").lower()
            plt.savefig(f'figures/{safe_title}_cas_dc.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(f'figures/{safe_title}_cas_dc.png', dpi=300, bbox_inches='tight')
        
        plt.show()