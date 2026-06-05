# python/src/plotting.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_dashboard(sim, approach_name: str, test_day: int, layout: str = 'grid', legend_loc: str = 'best'):
    """
    Generates a publication-ready dashboard for both Hybrid and Baseline strategies.
    
    Parameters:
        sim: The Simulator or HybridSimulator object.
        approach_name: Title of the evaluated strategy.
        test_day: The day of the dataset being evaluated.
        layout: 'grid' (2x2 dashboard) or 'separate' (individual A4 figures).
        legend_loc: Matplotlib legend location identifier (default: 'best').
    """
    is_hybrid = hasattr(sim, 'soc_history')
    num_plots = 4 if is_hybrid else 3
    
    # --- Refined Scientific Palette ---
    # Graph 1 (Power)
    c_demand   = '#08519c'   # Deep Navy (Professional, analytical)
    c_raw      = '#6baed6'   # Soft Light Gray (Background context)
    
    # Graph 2 (FC Ops)
    c_fc_pwr   = '#1d91c0'   # Bright Blue 
    c_mod_pw   = '#41b6c4'   # Soft Blue
    c_module   = '#fc8d59'   # Red-Orange (High contrast for step counts)
    
    # Graph 3 (Economics - Monochromatic/Gradient Logic)
    c_h2_cst   = '#e31a1c'   # Teal
    c_sw_cst   = '#ff7f00'   # Darker Teal
    c_bat_co   = '#1f78b4'   # Gray
    c_total    = '#000000'   # Sharp Black (Emphasis)
    
    # Graph 4 (Battery - High Pop for SoC)
    c_soc      = "#16568a" 
    c_rechr    = '#4daf4a'  # Green for positive power
    c_disch    = '#e41a1c'  # Red for negative power

    # --- Time Axis Conversion (Hours) ---
    t_raw_h = sim.raw_t / 3600.0
    if is_hybrid:
        t_plot_h = sim.t_micro / 3600.0
        t_macro_h = sim.t_micro[::sim.config.lambda_scale][:len(sim.C_o_vec)] / 3600.0
    else:
        t_plot_h = sim.t_macro / 3600.0
        t_macro_h = t_plot_h

    # --- Typography & Rendering Constraints ---
    rc_params = {
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.titlesize': 13,
        'font.family': 'sans-serif',
        'figure.dpi': 600
    }

    with plt.rc_context(rc_params):
        figs, axes = [], []
        if layout == 'grid':
            fig, axs = plt.subplots(2, 2, figsize=(14.0, 10.0), layout='constrained')
            figs.append(fig)
            axes = axs.flatten().tolist()
            if not is_hybrid:
                axes[3].set_visible(False) 
        elif layout == 'separate':
            a4_width = 7.0 
            for _ in range(num_plots):
                fig, ax = plt.subplots(figsize=(a4_width, 4.0), layout='constrained')
                figs.append(fig)
                axes.append(ax)
        else:
            raise ValueError("Layout must be 'grid' or 'separate'")

        # Calculate Total Voyage Cost
        if is_hybrid:
            total_cost = np.sum(sim.C_o_vec) + np.sum(sim.C_s_vec) + np.sum(sim.C_bat_vec)
        else:
            total_cost = np.sum(sim.C_o) + np.sum(sim.C_s)
        title_str = f"Strategy: {approach_name} | Total Cost: ${total_cost:,.2f} | Day: {test_day:02d}"
        
        if layout == 'grid' and figs:
            figs[0].suptitle(title_str, fontweight='bold')

        # =========================================================================
        # PLOT 1: Power Demand
        # =========================================================================
        ax1 = axes[0]
        ax1.set_title(f"{title_str}\nPower Demand" if layout == 'separate' else "Power Demand", fontweight='bold')

        ax1.plot(t_raw_h, sim.raw_pd, color=c_raw, linewidth=0.8, alpha=1.0, label='Raw Demand')
        ax1.plot(t_plot_h, sim.P_d, color=c_demand, linewidth=1.2, label='Filtered Demand')
            
        ax1.set_ylabel("Power [kW]", fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.legend(loc=legend_loc)

        # =========================================================================
        # PLOT 2: Fuel Cell Operations (Fixed Overlapping)
        # =========================================================================
        ax2 = axes[1]
        ax2.set_title(f"{title_str}\nFuel Cell Operations" if layout == 'separate' else "Fuel Cell Operations", fontweight='bold')
        
        fc_pwr = sim.pfc_history if is_hybrid else sim.P_d
        n_plot = sim.n_history if is_hybrid else sim.n
        
        n_safe = np.maximum(n_plot, 1)
        per_module_pwr = fc_pwr / n_safe
        
        # Total power is a solid anchor line
        ax2.plot(t_plot_h, fc_pwr, color=c_fc_pwr, linewidth=1.2, label='Total FC Power')
        
        ax2.plot(t_plot_h, per_module_pwr, color=c_mod_pw, linewidth=0.8, label='Power per Module (P/n)')
        ax2.set_ylabel("Power [kW]", fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.5)
        
        ax2_twin = ax2.twinx()
        ax2_twin.step(t_plot_h, n_plot, color=c_module, linewidth=1.5, where='post', linestyle='--', label='Active Modules (n)')
        ax2_twin.set_ylabel("Active Modules", color=c_module, fontweight='bold')
        ax2_twin.tick_params(axis='y', labelcolor=c_module)
        ax2_twin.set_ylim(np.min(sim.config.n_vals) - 1, np.max(sim.config.n_vals) + 1)
        
        l1, lab1 = ax2.get_legend_handles_labels()
        l2, lab2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(l1 + l2, lab1 + lab2, loc=legend_loc)

        # =========================================================================
        # PLOT 3: Cumulative Economic Trajectories
        # =========================================================================
        ax3 = axes[2]
        ax3.set_title(f"{title_str}\nEconomics" if layout == 'separate' else "Cumulative Costs", fontweight='bold')

        cum_o = np.cumsum(sim.C_o_vec) if is_hybrid else np.cumsum(sim.C_o)
        cum_s = np.cumsum(sim.C_s_vec) if is_hybrid else np.cumsum(sim.C_s)
        
        ax3.plot(t_macro_h, cum_o, color=c_h2_cst, linewidth=1.5, label='H2 Consumption')
        ax3.plot(t_macro_h, cum_s, color=c_sw_cst, linewidth=1.5, label='Switching Deg.')
        
        if is_hybrid:
            cum_bat = np.cumsum(sim.C_bat_vec)
            ax3.plot(t_macro_h, cum_bat, color=c_bat_co, linewidth=1.5, label='Battery Deg.')
            cum_total = cum_o + cum_s + cum_bat
        else:
            cum_total = cum_o + cum_s
            
        ax3.plot(t_macro_h, cum_total, color=c_total, linewidth=2.2, label='Total Cost')
        ax3.set_ylabel("Cumulative Cost [$]", fontweight='bold')
        ax3.grid(True, linestyle=':', alpha=0.5)
        ax3.legend(loc=legend_loc) 

        # =========================================================================
        # PLOT 4: Battery Operations (Conditional Coloring)
        # =========================================================================
        if is_hybrid:
            ax4 = axes[3]
            ax4.set_title(f"{title_str}\nBattery Operations" if layout == 'separate' else "Battery SoC & Power", fontweight='bold')
            
            # 1. Setup SoC Axis (Primary)
            ax4.set_ylabel("SoC [%]", color=c_soc, fontweight='bold')
            ax4.tick_params(axis='y', labelcolor=c_soc)
            ax4.set_ylim(0, 100)
            ax4.grid(True, linestyle=':', alpha=0.5, zorder=1)
            ax4.patch.set_visible(False) 
            
            # 2. Setup Power Axis (Twin)
            pbat = sim.pbat_history
            ax4_twin = ax4.twinx()
            
            # Ensure 0 kW (Power) aligns with 50% (SoC)
            max_abs_p = np.max(np.abs(pbat))
            limit = max_abs_p * 1.2 if max_abs_p > 0 else 100
            
            ax4_twin.set_ylim(-limit, limit)
            ax4_twin.set_ylabel("Battery Power [kW]", fontweight='bold')
            
            # Z-Order layering
            ax4_twin.set_zorder(0)
            ax4.set_zorder(1)
            ax4_twin.patch.set_visible(False)

            # 3. Vectorized Line Coloring (Alpha reduced to 0.6)
            points = np.array([t_plot_h, pbat]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = [c_disch if p >= 0 else c_rechr for p in pbat]
            
            lc = LineCollection(segments, colors=colors, linewidth=0.8, alpha=0.5, clip_on=True)
            ax4_twin.add_collection(lc)

            # 4. Fills (Alpha reduced to 0.15)
            ax4_twin.fill_between(t_plot_h, 0, pbat, where=(pbat >= 0), facecolor=c_disch, alpha=0.3, interpolate=True, clip_on=True)
            ax4_twin.fill_between(t_plot_h, 0, pbat, where=(pbat < 0), facecolor=c_rechr, alpha=0.3, interpolate=True, clip_on=True)

            # 5. Horizontal Reference Anchor & SoC Curve
            ax4.axhline(50.0, color="#000000", linestyle='--', linewidth=1.5, zorder=2)
            ax4.plot(t_plot_h, sim.soc_history[:-1], color=c_soc, linewidth=3, label='State of Charge (SoC)', zorder=3)
            
            # 6. Legend
            from matplotlib.lines import Line2D
            custom_handles = [
                Line2D([0], [0], color=c_soc, lw=3, label='State of Charge (SoC)'),
                Line2D([0], [0], color=c_disch, lw=0.8, label='Discharging'),
                Line2D([0], [0], color=c_rechr, lw=0.8, label='Recharging')
            ]
            ax4.legend(handles=custom_handles, loc=legend_loc)

        plt.show()