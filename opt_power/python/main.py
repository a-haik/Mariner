# python/main.py
import os
import numpy as np
from config import SimConfig
from src.data_processing import load_and_interpolate_sov_data, calibrate_markov_chain, downsample_block_mean
from src.simulator import Simulator
from src.controllers.constant import ConstantControl
from src.controllers.threshold import ThresholdControl
from src.controllers.stochastic import StochasticControl
from src.plotting import plot_costs_and_control, plot_cost_comparison

def main():
    print("=" * 70)
    print("MARINER CO-PILOT: Initiating Optimal Power Porting Pipeline (Numba)")
    print("=" * 70)
    
    # 1. Initialize the unified configuration
    config = SimConfig()
    
    # 2. Define training datasets to calibrate the Markov model
    # Paths assume you place the .mat files within the project's data/ folder
    training_files = [
        '../data/SOV_05-Feb-2023.mat',
        '../data/SOV_06-Feb-2023.mat',
        '../data/SOV_07-Feb-2023.mat',
        '../data/SOV_08-Feb-2023.mat',
        '../data/SOV_09-Feb-2023.mat',
        '../data/SOV_10-Feb-2023.mat'
    ]
    
    # Check if data files are present before executing calibration loop
    # If the data/ folder is empty due to confidentiality, we catch it gracefully
    for f in training_files:
        if not os.path.exists(f):
            print(f"\n[CRITICAL WARNING] File not found: {f}")
            print("Please ensure your supervisor's confidential vessel files are located in the 'data/' directory.")
            print("Aborting optimization run.\n")
            return

    print("\nExecuting Step 3: Loading and calibrating Markov chain from SOV telemetry...")
    raw_training_data = load_and_interpolate_sov_data(training_files)
    mc_model = calibrate_markov_chain(raw_training_data, config)
    print(f"-> Calibration successful. State space levels derived: {config.n_states} tracking zones.")
    
    # 3. Extract the evaluation profile trajectory (isolated verification day)
    print("\nExecuting Step 4: isolating validation tracking trajectory (Day: 08-Feb)...")
    validation_file = ['../data/SOV_08-Feb-2023.mat']
    raw_validation_data = load_and_interpolate_sov_data(validation_file)
    
    # Downsample validation demand trend to the macro tracking resolution window Ts
    ds_validation = downsample_block_mean(raw_validation_data['t'], raw_validation_data['Pd'], config.Ts, align='t0')
    validation_Pd = ds_validation['Pd'] # Flattened tracking demand array
    
    # 4. Instantiate the Numba-accelerated controller instances
    print("\nExecuting Step 5: Instantiating control laws and triggering Numba JIT compilation...")
    controllers = [
        ConstantControl(),
        StochasticControl(
            k_s=config.k_s, 
            p_star=config.p_star, 
            states=mc_model['levels'], 
            transition_matrix=mc_model['P'], 
            n_vals=config.n_vals
        ),
        ThresholdControl(config)
    ]
    
    controller_names = [
        "ConstantControl",
        "StochasticControl",
        "ThresholdControl"
    ]
    
    # 5. Spin up simulator plant instances and run benchmarks
    print("\nExecuting Step 6: Driving evaluation loops across plant simulators...")
    simulators = []
    for idx, ctrl in enumerate(controllers):
        sim = Simulator(config, validation_Pd)
        total_cost = sim.run(ctrl)
        simulators.append(sim)
        
        print(f"-> {controller_names[idx]:<18} | Total Cost: {total_cost:10.2f} | Op Cost: {np.sum(sim.C_o):10.2f} | Switch Cost: {np.sum(sim.C_s):10.2f}")
        
    # 6. Export comparative performance charts
    print("\nSaving performance summaries to 'figures/' workspace directory...")
    plot_costs_and_control(simulators, controller_names, validation_Pd)
    plot_cost_comparison(simulators, controller_names)
    print("-> Visualization complete.")
    
    # ==============================================================================
    # STEP 7: LOCAL NUMERICAL EQUIVALENCE CHECK (Mentoring Assertion Loop)
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 7: NUMERICAL EQUIVALENCE VERIFICATION DIAGNOSTIC")
    print("=" * 70)
    print("To confirm your Python refactor yields the exact same results as MATLAB:")
    print("1. Run your supervisor's code in MATLAB for Day 08.")
    print("2. Verify that the output prints match up to your local terminal results above.")
    print("Expected benchmark reference limits for Stochastic Control:")
    print("   Total Operational Cost around 39.00 | Total Switching Cost around 2.00")
    print("=" * 70)

if __name__ == '__main__':
    main()