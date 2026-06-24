# python/main.py
import os
from config import SimConfig
from src.data_processing import load_and_interpolate_sov_data, calibrate_markov_chain, downsample_block_mean
from src.simulator import Simulator
from src.plants.fc_only_plant import FuelCellOnlyPlant
from src.plants.hybrid_plant import HybridPlant
from src.solvers.sdp_baseline import BaselineSDPSolver
from src.controllers.constant import ConstantControl
from src.controllers.threshold import ThresholdControl
from src.controllers.stochastic import StochasticControl
from src.plotting import plot_dynamic_history, plot_cost_comparison

def main():
    print("=" * 70)
    print("MARINER CO-PILOT: Decoupled Continuous-Time Execution Pipeline")
    print("=" * 70)
    
    config = SimConfig()
    
    training_files = [
        '../data/SOV_05-Feb-2023.mat', '../data/SOV_06-Feb-2023.mat', 
        '../data/SOV_07-Feb-2023.mat', '../data/SOV_08-Feb-2023.mat', 
        '../data/SOV_09-Feb-2023.mat', '../data/SOV_10-Feb-2023.mat'
    ]
    
    for f in training_files:
        if not os.path.exists(f):
            print(f"\n[CRITICAL WARNING] File not found: {f}")
            print("Please ensure your supervisor's confidential vessel files are located in the 'data/' directory.")
            return

    print("\nExecuting Step 3: Loading and calibrating Markov chain from SOV telemetry...")
    raw_training_data = load_and_interpolate_sov_data(training_files)
    mc_model = calibrate_markov_chain(raw_training_data, config)
    print(f"-> Calibration successful. State space levels derived: {config.n_states} tracking zones.")
    
    print("\nExecuting Step 4: Isolating validation tracking trajectory (Day: 08-Feb)...")
    validation_file = ['../data/SOV_08-Feb-2023.mat']
    raw_validation_data = load_and_interpolate_sov_data(validation_file)
    P_d_continuous = raw_validation_data['Pd']
    
    # We downsample strictly to determine the backward induction horizon length for the offline solver
    ds_validation = downsample_block_mean(raw_validation_data['t'], P_d_continuous, config.Ts, align='t0')
    horizon_length = len(ds_validation['Pd'])
    
    # NOTE: You can easily swap this out for `HybridPlant(config)` when you're ready
    plant = FuelCellOnlyPlant(config)
    
    print("\nTriggering Offline Bellman Solver Induction Loops...")
    sdp_solver = BaselineSDPSolver(config, mc_model)
    baseline_policy_matrix = sdp_solver.compute_policy_matrix(horizon_length=horizon_length)
    
    print("\nInstantiating control laws...")
    controllers = [
        ConstantControl(config),
        StochasticControl(
            states=mc_model['levels'],
            n_vals=config.n_vals,
            policy_matrix=baseline_policy_matrix
        ),
        ThresholdControl(config=config, horizon_length=horizon_length, sigma=0.5)
    ]
    
    controller_names = ["ConstantControl", "StochasticControl", "ThresholdControl"]
    
    print("\nExecuting Step 6: Driving evaluation loops across ZOH plant simulators...")
    simulators = []
    for idx, ctrl in enumerate(controllers):
        sim = Simulator(config, P_d_continuous, plant)
        total_cost = sim.run(ctrl)
        simulators.append(sim)
        print(f"-> {controller_names[idx]:<18} | Total Cost: {total_cost:10.2f} €")
        
    print("\nSaving performance summaries to 'figures/' workspace directory...")
    plot_dynamic_history(simulators, controller_names, save_plot=True)
    plot_cost_comparison(simulators, controller_names, save_plot=True)
    print("-> Visualization complete. You are clear to begin hybrid battery development.")

if __name__ == '__main__':
    main()