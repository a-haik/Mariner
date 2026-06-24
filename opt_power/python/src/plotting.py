# python/src/plotting.py
import os
import numpy as np
import pandas as pd
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

def plot_benchmarker_results(df: pd.DataFrame, title: str, plot_type: str = 'bar'):
    """
    Visualizes the outputs from the VoyageBenchmarker.
    
    Parameters:
        df: The pandas DataFrame returned by the benchmarker.
        title: The title of the plot.
        plot_type: 'bar' for Leave-One-Out comparisons, 'line' for Forward Chaining trends.
    """
    import matplotlib.pyplot as plt
    import os
    
    # 1. Separate the "Average" row from the individual runs
    if 'Average' in df.index:
        df_runs = df.drop('Average')
        avg_cost = df.loc['Average', 'Total Cost [€]']
    else:
        df_runs = df
        avg_cost = None

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 2. Extract just the Total Cost for clarity
    costs = df_runs['Total Cost [€]']
    x_labels = df_runs.index

    # 3. Plotting Logic
    if plot_type == 'bar':
        ax.bar(x_labels, costs, color='royalblue', alpha=0.8, edgecolor='black')
        ax.set_ylabel("Total Cost [€]")
        plt.xticks(rotation=45, ha='right')
        
    elif plot_type == 'line':
        ax.plot(x_labels, costs, marker='o', color='darkorange', linewidth=2, markersize=8)
        ax.set_ylabel("Total Cost [€]")
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    # 4. Add the Average Benchmark Line
    if avg_cost is not None:
        ax.axhline(avg_cost, color='red', linestyle='--', linewidth=1.5, label=f'Average Cost: {avg_cost:.2f} €')
        ax.legend()

    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    
    # Save and display
    os.makedirs('figures', exist_ok=True)
    safe_title = title.replace(" ", "_").lower()
    plt.savefig(f'figures/{safe_title}.png', dpi=300)
    plt.show()