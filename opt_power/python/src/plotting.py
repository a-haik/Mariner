# python/src/plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt
from src.simulator import Simulator

def plot_costs_and_control(simulators: list[Simulator], controller_names: list[str], P_d: np.ndarray):
    """
    Generates a 3 x NumControllers subplot grid tracking time-series performance metrics.
    Perfect replication of plot_costs_and_control in plotting_functions.m.
    Resolves physical scale differences between Power [kW] and Module Counts using twinx().
    """
    num_controllers = len(simulators)
    fig, axes = plt.subplots(3, num_controllers, figsize=(6 * num_controllers, 12), sharex='row')
    
    if num_controllers == 1:
        axes = np.expand_dims(axes, axis=1)
        
    # Calculate global tracking bounds across all cost simulation datasets
    c_o_min = min([np.min(sim.C_o) for sim in simulators])
    c_o_max = max([np.max(sim.C_o) for sim in simulators])
    c_s_min = min([np.min(sim.C_s) for sim in simulators])
    c_s_max = max([np.max(sim.C_s) for sim in simulators])
    p_min, p_max = np.min(P_d), np.max(P_d)
    
    # --- Calculate uniform control tracking limits across all models ---
    # Lower limit: matches the absolute lowest n value plotted across all controllers
    n_min_global = min([np.min(sim.n) for sim in simulators])
    # Upper limit: matches the highest n possible defined in the action space configuration (10)
    n_max_possible = max([np.max(sim.config.n_vals) for sim in simulators])
    
    for idx in range(num_controllers):
        sim = simulators[idx]
        name = controller_names[idx]
        
        # Row 1: Operating Cost Trajectories
        axes[0, idx].plot(sim.C_o, label='Operating Cost')
        axes[0, idx].set_title(f"{name}\nOperating Cost")
        axes[0, idx].set_ylabel("Cost")
        axes[0, idx].set_ylim(c_o_min, c_o_max)
        axes[0, idx].grid(True)
        
        # Row 2: Switching Cost Trajectories
        axes[1, idx].plot(sim.C_s, color='darkorange', label='Switching Cost')
        axes[1, idx].set_title("Switching Cost")
        axes[1, idx].set_ylabel("Cost")
        axes[1, idx].set_ylim(c_s_min, c_s_max + 1.0)
        axes[1, idx].grid(True)
        
        # Row 3: Load Demand vs Active Module Allocation (Dual-Scale Axis Layout)
        # Primary Left Axis: Power Demand in kW
        axes[2, idx].plot(P_d, color='lime', label='Power Demand')
        axes[2, idx].set_title("Demand vs Control")
        axes[2, idx].set_xlabel("Time Step")
        axes[2, idx].set_ylabel("Power Demand [kW]", color='darkgreen')
        axes[2, idx].set_ylim(p_min / 1.1, p_max * 1.1)
        axes[2, idx].tick_params(axis='y', labelcolor='darkgreen')
        axes[2, idx].grid(True)
        
        # Secondary Right Axis: Twin layout to display the module allocation trajectory
        ax_n = axes[2, idx].twinx()
        ax_n.plot(sim.n, color='fuchsia', label='Control (n)', linewidth=1.5)
        ax_n.set_ylabel("Active Modules (n)", color='fuchsia')
        
        # Apply your exact constraint parameters
        ax_n.set_ylim(n_min_global-1, n_max_possible)
        ax_n.tick_params(axis='y', labelcolor='fuchsia')
        
        # Merge legends from both the left and right axes cleanly onto a single legend box
        lines_left, labels_left = axes[2, idx].get_legend_handles_labels()
        lines_right, labels_right = ax_n.get_legend_handles_labels()
        ax_n.legend(lines_left + lines_right, labels_left + labels_right, loc='upper left')
        
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/costs_and_control.png', dpi=300)
    plt.close()


def plot_cost_comparison(simulators: list[Simulator], controller_names: list[str]):
    """
    Generates a grouped bar chart displaying cumulative operational and switching costs.
    Perfect replication of plot_cost_comparison in main.m.
    """
    total_op_costs = [np.sum(sim.C_o) for sim in simulators]
    total_switch_costs = [np.sum(sim.C_s) for sim in simulators]
    
    x = np.arange(len(controller_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, total_op_costs, width, label='Operational Cost')
    rects2 = ax.bar(x + width/2, total_switch_costs, width, label='Switching Cost')
    
    ax.set_ylabel('Total Cost')
    ax.set_title('Total Operational and Switching Costs by Controller')
    ax.set_xticks(x)
    ax.set_xticklabels(controller_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Text label formatting on top of each bar
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/cost_comparison.png', dpi=300)
    plt.close()