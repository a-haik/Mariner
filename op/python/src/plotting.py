# python/src/plotting.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_dashboard(sim, approach_name: str, test_day: int, layout: str = 'grid', legend_loc: str = 'best'):
    """
    Generates a universal 4-panel publication-ready dashboard.
    All controllers are now evaluated in the hybrid physical space.
    """
    # --- Time Axis Conversion (Hours) ---
    t_raw_h = sim.raw_t / 3600.0
    t_plot_h = sim.t_micro / 3600.0
    t_macro_h = sim.t_micro[::sim.config.lambda_scale][:len(sim.C_o_vec)] / 3600.0

    # --- Typography & Rendering Constraints ---
    rc_params = {
        'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 8, 'figure.titlesize': 13,
        'font.family': 'sans-serif', 'figure.dpi': 600
    }

    # Colors remain identical to your previous script...
    c_demand, c_raw = '#08519c', '#6baed6'
    c_fc_pwr, c_mod_pw, c_module = '#1d91c0', '#41b6c4', '#fc8d59'
    c_h2_cst, c_sw_cst, c_bat_co, c_total = '#e31a1c', '#ff7f00', '#1f78b4', '#000000'
    c_soc, c_rechr, c_disch = "#16568a", '#4daf4a', '#e41a1c'

    with plt.rc_context(rc_params):
        if layout == 'grid':
            fig, axs = plt.subplots(2, 2, figsize=(14.0, 10.0), layout='constrained')
            axes = axs.flatten().tolist()
        elif layout == 'separate':
            axes = [plt.subplots(figsize=(7.0, 4.0), layout='constrained')[1] for _ in range(4)]

        # 1. Title & Costs
        total_cost = np.sum(sim.C_o_vec) + np.sum(sim.C_s_vec) + np.sum(sim.C_bat_vec)
        # Check if penalty wall was hit to add a warning to the title
        penalty_flag = " [! BOUNDS VIOLATED !]" if total_cost >= sim.config.penalty_wall else ""
        title_str = f"Strategy: {approach_name} | Total Cost: ${total_cost:,.2f}{penalty_flag} | Day: {test_day:02d}"
        
        if layout == 'grid':
            fig.suptitle(title_str, fontweight='bold', color='red' if penalty_flag else 'black')

        # --- PLOT 1: Power Demand ---
        axes[0].set_title(title_str + "\nPower Demand" if layout == 'separate' else "Power Demand", fontweight='bold')
        axes[0].plot(t_raw_h, sim.raw_pd, color=c_raw, linewidth=0.8, alpha=1.0, label='Raw Demand')
        axes[0].plot(t_plot_h, sim.P_d, color=c_demand, linewidth=1.2, label='Filtered Demand')
        axes[0].set_ylabel("Power [kW]", fontweight='bold')
        axes[0].grid(True, linestyle=':', alpha=0.5)
        axes[0].legend(loc=legend_loc)

        # --- PLOT 2: Fuel Cell Operations ---
        axes[1].set_title(title_str + "\nFuel Cell Operations" if layout == 'separate' else "Fuel Cell Operations", fontweight='bold')
        n_safe = np.maximum(sim.n_history, 1)
        per_module_pwr = sim.pfc_history / n_safe
        
        axes[1].plot(t_plot_h, per_module_pwr, color=c_mod_pw, linewidth=1.2, label='Power per Module (P/n)')
        axes[1].set_ylabel("Power [kW]", fontweight='bold')
        axes[1].grid(True, linestyle=':', alpha=0.5)
        
        ax2_twin = axes[1].twinx()
        ax2_twin.step(t_plot_h, sim.n_history, color=c_module, linewidth=1.8, linestyle='--', where='post', label='Active Modules (n)')
        ax2_twin.set_ylabel("Active Modules", color=c_module, fontweight='bold')
        ax2_twin.set_ylim(np.min(sim.config.n_vals) - 1, np.max(sim.config.n_vals) + 1)
        axes[1].plot(t_plot_h, sim.pfc_history, color=c_fc_pwr, linewidth=1.2, label='Total FC Power')
        
        l1, lab1 = axes[1].get_legend_handles_labels()
        l2, lab2 = ax2_twin.get_legend_handles_labels()
        axes[1].legend(l1 + l2, lab1 + lab2, loc=legend_loc)

        # --- PLOT 3: Cumulative Economics ---
        axes[2].set_title(title_str + "\nEconomics" if layout == 'separate' else "Cumulative Costs", fontweight='bold')
        cum_o, cum_s, cum_bat = np.cumsum(sim.C_o_vec), np.cumsum(sim.C_s_vec), np.cumsum(sim.C_bat_vec)
        
        axes[2].plot(t_macro_h, cum_o, color=c_h2_cst, linewidth=1.5, label='H2 Consumption')
        axes[2].plot(t_macro_h, cum_s, color=c_sw_cst, linewidth=1.5, label='Switching Deg.')
        axes[2].plot(t_macro_h, cum_bat, color=c_bat_co, linewidth=1.5, label='Battery Deg. (incl. penalties)')
        axes[2].plot(t_macro_h, cum_o + cum_s + cum_bat, color=c_total, linewidth=2.2, label='Total Cost')
        
        # If penalty wall is hit, scale the Y-axis to see the normal costs, otherwise it flattens everything
        if total_cost >= sim.config.penalty_wall:
            axes[2].set_ylim(0, np.max(cum_o + cum_s) * 1.5)
            
        axes[2].set_ylabel("Cumulative Cost [$]", fontweight='bold')
        axes[2].grid(True, linestyle=':', alpha=0.5)
        axes[2].legend(loc=legend_loc) 

        # --- PLOT 4: Battery Operations ---
        axes[3].set_title(title_str + "\nBattery Operations" if layout == 'separate' else "Battery SoC & Power", fontweight='bold')
        axes[3].set_ylabel("SoC [%]", color=c_soc, fontweight='bold')
        
        # Let the SoC scale naturally to show deficit
        miN_s, max_soc = np.min(sim.soc_history), np.max(sim.soc_history)
        axes[3].set_ylim(min(0, miN_s * 1.1), max(100, max_soc * 1.1))
        
        axes[3].grid(True, linestyle=':', alpha=0.5, zorder=1)
        axes[3].patch.set_visible(False) 
        
        pbat = sim.pbat_history
        ax4_twin = axes[3].twinx()
        limit = np.max(np.abs(pbat)) * 1.2 if np.max(np.abs(pbat)) > 0 else 100
        ax4_twin.set_ylim(-limit, limit)
        ax4_twin.set_ylabel("Battery Power [kW]", fontweight='bold')
        ax4_twin.set_zorder(0)
        axes[3].set_zorder(1)
        ax4_twin.patch.set_visible(False)

        # Plot SoC limits
        axes[3].axhline(100.0, color="red", linestyle='-', linewidth=1.0, alpha=0.5, zorder=2)
        axes[3].axhline(0.0, color="red", linestyle='-', linewidth=1.0, alpha=0.5, zorder=2)
        axes[3].axhline(50.0, color="#000000", linestyle='--', linewidth=1.5, zorder=2)
        
        axes[3].plot(t_plot_h, sim.soc_history[:-1], color=c_soc, linewidth=3, label='State of Charge (SoC)', zorder=3)
        
        # Power fill coloring
        ax4_twin.fill_between(t_plot_h, 0, pbat, where=(pbat >= 0), facecolor=c_disch, alpha=0.3, interpolate=True, clip_on=True)
        ax4_twin.fill_between(t_plot_h, 0, pbat, where=(pbat < 0), facecolor=c_rechr, alpha=0.3, interpolate=True, clip_on=True)

        from matplotlib.lines import Line2D
        custom_handles = [
            Line2D([0], [0], color=c_soc, lw=3, label='State of Charge (SoC)'),
            Line2D([0], [0], color=c_disch, lw=0.8, label='Discharging'),
            Line2D([0], [0], color=c_rechr, lw=0.8, label='Recharging')
        ]
        axes[3].legend(handles=custom_handles, loc=legend_loc)

        plt.show()