# python/main.py
import os
import numpy as np
from config import SimConfig
from src.data_processing import load_and_interpolate_sov_data, calibrate_markov_chain, downsample_block_mean
from src.simulator import Simulator
from src.plants.fc_only_plant import FuelCellOnlyPlant
from opt_power.python.src.solvers.sdp_baseline import BaselineSDPSolver
from src.controllers.constant import ConstantControl
from src.controllers.threshold import ThresholdControl
from src.controllers.stochastic import StochasticControl
from src.plotting import plot_costs_and_control, plot_cost_comparison

def main():
    print("=" * 70)
    print("MARINER CO-PILOT: Initiating Decoupled Power Distribution Pipeline")
    print("=" * 70)
    
    # 1. Initialize the unified configuration
    config = SimConfig()
    
    # 2. Define training datasets to calibrate the Markov model
    training_files = [
        '../data/SOV_05-Feb-2023.mat',
        '../data/SOV_06-Feb-2023.mat',
        '../data/SOV_07-Feb-2023.mat',
        '../data/SOV_08-Feb-2023.mat',
        '../data/SOV_09-Feb-2023.mat',
        '../data/SOV_10-Feb-2023.mat'
    ]
    
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
    
    ds_validation = downsample_block_mean(raw_validation_data['t'], raw_validation_data['Pd'], config.Ts, align='t0')
    validation_Pd = ds_validation['Pd']
    
    # 4. Instantiate the Physical Asset (The Plant Environment)
    plant = FuelCellOnlyPlant(config)
    
    # 5. Compute the Offline SDP Policy Matrix BEFORE building the controller
    print("\nTriggering Offline Bellman Solver Induction Loops...")
    sdp_solver = BaselineSDPSolver(config, mc_model)
    baseline_policy_matrix = sdp_solver.compute_policy_matrix(horizon_length=len(validation_Pd))
    
    # 6. Instantiate the controllers (Injecting the solved policy matrix into StochasticControl)
    print("\nInstantiating control laws...")
    controllers = [
        ConstantControl(),
        StochasticControl(
            states=mc_model['levels'], 
            n_vals=config.n_vals, 
            policy_matrix=baseline_policy_matrix
        ),
        ThresholdControl(config)
    ]
    
    controller_names = [
        "ConstantControl",
        "StochasticControl",
        "ThresholdControl"
    ]
    
    # 7. Spin up simulator plant instances and run benchmarks
    print("\nExecuting Step 6: Driving evaluation loops across plant simulators...")
    simulators = []
    for idx, ctrl in enumerate(controllers):
        # Pass the unique plant instance directly into our generic simulation wrapper
        sim = Simulator(config, validation_Pd, plant)
        total_cost = sim.run(ctrl)
        simulators.append(sim)
        
        print(f"-> {controller_names[idx]:<18} | Total Cost: {total_cost:10.2f} | Op Cost: {np.sum(sim.C_o):10.2f} | Switch Cost: {np.sum(sim.C_s):10.2f}")
        
    # 8. Export comparative performance charts
    print("\nSaving performance summaries to 'figures/' workspace directory...")
    plot_costs_and_control(simulators, controller_names, validation_Pd)
    plot_cost_comparison(simulators, controller_names)
    print("-> Visualization complete.")

if __name__ == '__main__':
    main()