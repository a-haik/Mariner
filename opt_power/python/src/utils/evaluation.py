# python/src/utils/evaluation.py
import time
import numpy as np
import pandas as pd
from src.data_processing import fit_dtmc, downsample_block_mean
from src.solvers.sdp_baseline import BaselineSDPSolver
from src.solvers.sdp_hybrid import HybridSDPSolver
from src.controllers.threshold import ThresholdControl
from src.controllers.stochastic import StochasticControl
from src.controllers.hybrid_heuristic import HybridThresholdControl
from src.controllers.hybrid_stochastic import HybridStochasticControl
from src.simulator import Simulator, HybridSimulator

class VoyageBenchmarker:
    """
    Automated evaluation engine for Cross-Validation and Benchmarking.
    Handles data isolation, model training, and metric extraction.
    """
    def __init__(self, fleet_cache: dict, exclude_days: list = None):
        self.fleet_cache = fleet_cache
        self.exclude_days = exclude_days or []
        # Sort all valid days to ensure chronological integrity
        self.valid_days = sorted([d for d in fleet_cache.keys() if d not in self.exclude_days])

    def _prepare_data(self, train_days: list, test_day: int) -> tuple[dict, dict]:
        """Safely concatenates training days and isolates the test day."""
        train_t, train_pd = [], []
        for d in train_days:
            data = self.fleet_cache[d]
            t_off = train_t[-1][-1] if train_t else 0.0
            train_t.append(data['t'] + t_off)
            train_pd.append(data['Pd'])

        train_data = {'t': np.concatenate(train_t), 'Pd': np.concatenate(train_pd)}
        test_data = self.fleet_cache[test_day]
        return train_data, test_data

    def _evaluate_single_run(self, approach: dict, train_data: dict, test_data: dict) -> dict:
        """Trains and evaluates a single approach configuration."""
        start_time = time.time()
        
        config = approach['config']
        plant = approach['plant']
        raw_t = test_data['t']       # Keep the raw 1Hz timestamps
        raw_pd = test_data['Pd']     # Keep the raw 1Hz demand
        
        # --- Micro/Macro Downsampling ---
        if approach['is_hybrid']:
            # Downsample 1Hz data to the micro-step (dt)
            ds_micro = downsample_block_mean(raw_t, raw_pd, config.dt, align='t0')
            test_target_pd = ds_micro['Pd']
            sim_t = ds_micro['t']
            macro_horizon = len(test_target_pd) // config.lambda_scale
        else:
            # Baseline requires explicitly downsampled macro-steps (Ts)
            ds_macro = downsample_block_mean(raw_t, raw_pd, config.Ts, align='t0')
            test_target_pd = ds_macro['Pd']
            sim_t = ds_macro['t']
            macro_horizon = len(test_target_pd)

        # --- 2. Controller Setup (Training) ---
        if approach['strategy'] == 'SDP':
            
            # Branch the DTMC timescale fitting based on the solver variant
            if approach['is_hybrid']:
                sdp_variant = approach.get('sdp_variant', 'MEAN_PROXY')
                if sdp_variant in ['EXACT_TREE', 'TENSOR_SWEEP']:
                    # Tensor/Exact approaches need micro-scale (dt) transition probabilities
                    ds_train = downsample_block_mean(train_data['t'], train_data['Pd'], config.dt, align='t0')
                else:
                    # Mean Proxy needs macro-scale transitions (dt * lambda_scale)
                    macro_step_sec = config.lambda_scale * config.dt
                    ds_train = downsample_block_mean(train_data['t'], train_data['Pd'], macro_step_sec, align='t0')
            else:
                # Baseline explicitly uses Ts (macro-step)
                ds_train = downsample_block_mean(train_data['t'], train_data['Pd'], config.Ts, align='t0')
                
            mc_model = fit_dtmc(ds_train['Pd'], config.n_states, config.alpha)
            
            if approach['is_hybrid']:
                solver = HybridSDPSolver(config, mc_model, variant=approach.get('sdp_variant', 'MEAN_PROXY'))
                pol_n, pol_pfc = solver.compute_policy_tensors(macro_horizon)
                controller = HybridStochasticControl(config, mc_model['levels'], pol_n, pol_pfc)
            else:
                solver = BaselineSDPSolver(config, mc_model)
                policy = solver.compute_policy_matrix(macro_horizon)
                controller = StochasticControl(mc_model['levels'], config.n_vals, policy)
                
        elif approach['strategy'] == 'HEURISTIC':
            if approach['is_hybrid']:
                controller = HybridThresholdControl(config, macro_horizon, tau_relax=5.0)
            else:
                controller = ThresholdControl(config)
        else:
            raise ValueError(f"Unknown strategy: {approach['strategy']}")

        # --- 3. Simulator Execution & Data Plumbing ---
        if approach['is_hybrid']:
            sim = HybridSimulator(config, test_target_pd, plant)
            sim.t_micro = sim_t  # Attach time array for plotting
        else:
            sim = Simulator(config, test_target_pd, plant)
            sim.t_macro = sim_t  # Attach time array for plotting
            
        # Inject the raw 1Hz background data into the simulator for plotting
        sim.raw_t = raw_t
        sim.raw_pd = raw_pd

        sim.run(controller)
        compute_time = time.time() - start_time

        # --- 4. Metric Extraction ---
        op_cost = np.sum(sim.C_o_vec) if approach['is_hybrid'] else np.sum(sim.C_o)
        sw_cost = np.sum(sim.C_s_vec) if approach['is_hybrid'] else np.sum(sim.C_s)
        bat_cost = np.sum(sim.C_bat_vec) if approach['is_hybrid'] else 0.0
        final_soc = sim.soc_history[-1] if approach['is_hybrid'] else np.nan

        return {
            "Total Cost ($)": op_cost + sw_cost + bat_cost,
            "H2 Cost ($)": op_cost,
            "FC Switch Cost ($)": sw_cost,
            "Bat. Degrade ($)": bat_cost,
            "Final SoC (%)": final_soc,
            "Compute Time (s)": compute_time,
            "simulator": sim
        }

    # =========================================================================
    # PUBLIC API: THE THREE EVALUATION MODES
    # =========================================================================

    def compare_approaches(self, approaches_dict: dict, train_days: list, test_day: int) -> pd.DataFrame:
        """Compares multiple approaches on the exact same train/test split"""
        print(f"\nComparing {len(approaches_dict)} approaches | Train: {train_days} | Test: Day {test_day}")
        train_data, test_data = self._prepare_data(train_days, test_day)
        
        results = {}
        for name, approach in approaches_dict.items():
            print(f" -> Running: {name}")
            results[name] = self._evaluate_single_run(approach, train_data, test_data)
            
        df = pd.DataFrame.from_dict(results, orient='index')
        sims = {name: res['simulator'] for name, res in results.items()}
        df = df.drop(columns=['simulator'])
        return df, sims

    def run_forward_chaining(self, approach: dict, min_train_days: int = 3) -> pd.DataFrame:
        """Chronological evaluation (Train 1..t-1, Test t)."""
        print(f"\nStarting Forward Chaining CV for: {approach.get('name', 'Model')}")
        results = {}
        
        for i in range(min_train_days, len(self.valid_days)):
            train_days = self.valid_days[:i]
            test_day = self.valid_days[i]
            print(f" -> Chaining Step {i - min_train_days + 1}: Training on {train_days} | Testing on {test_day}")
            
            train_data, test_data = self._prepare_data(train_days, test_day)
            metrics = self._evaluate_single_run(approach, train_data, test_data)
            metrics['Train Horizon'] = f"{len(train_days)} days"
            results[f"Test Day {test_day:02d}"] = metrics
            
        return pd.DataFrame.from_dict(results, orient='index')

    def run_leave_one_out(self, approach: dict) -> pd.DataFrame:
        """Leave-One-Out evaluation (Test on D, Train on all others)"""
        print(f"\nStarting Leave-One-Out CV for: {approach.get('name', 'Model')}")
        results = {}
        
        for test_day in self.valid_days:
            train_days = [d for d in self.valid_days if d != test_day]
            print(f" -> LOO Fold: Testing on {test_day} | Training on remaining {len(train_days)} days")
            
            train_data, test_data = self._prepare_data(train_days, test_day)
            metrics = self._evaluate_single_run(approach, train_data, test_data)
            results[f"Test Day {test_day:02d}"] = metrics
            
        return pd.DataFrame.from_dict(results, orient='index')