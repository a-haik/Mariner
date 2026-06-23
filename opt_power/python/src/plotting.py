# python/src/plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_dynamic_history(simulators, controller_names):
    """
    Dynamically generates subplots based on the keys present in the Simulator's history dictionary.
    Replaces the brittle, hardcoded plot_costs_and_control function.
    """
    if not simulators:
        return
        
    # Exclude time from the plotted variables
    history_keys = [k for k in simulators[0].history.keys() if k != 'time']
    num_controllers = len(simulators)
    num_metrics = len(history_keys)
    
    fig, axes = plt.subplots(num_metrics, num_controllers, figsize=(6 * num_controllers, 2.5 * num_metrics), sharex='col')
    
    # Matrix shape handling for single controller or metric situations
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
            
            # Use specific colors for better readability
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
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/dynamic_history.png', dpi=300)
    plt.close()

def plot_cost_comparison(simulators, controller_names):
    """
    Generates a grouped bar chart summarizing cumulative costs.
    """
    # Dynamically find all tracking variables starting with 'cost_'
    cost_keys = [k for k in simulators[0].history.keys() if k.startswith('cost_') and k != 'cost_total']
    
    x = np.arange(len(controller_names))
    width = 0.8 / len(cost_keys) if cost_keys else 0.5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, key in enumerate(cost_keys):
        # Accumulate the high-frequency 1Hz step costs
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
    ax.set_title('Cost Breakdown Comparison Across Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(controller_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/cost_comparison.png', dpi=300)
    plt.close()