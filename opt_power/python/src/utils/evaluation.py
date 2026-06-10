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
from src.simulator import HybridSimulator

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
        
        # 1. Universal Micro/Macro Downsampling
        ds_micro = downsample_block_mean(raw_t, raw_pd, config.dt, align='t0')
        test_target_pd = ds_micro['Pd']
        sim_t = ds_micro['t']
        
        # CRITICAL FIX: Force exact integer horizon alignment
        macro_horizon = len(test_target_pd) // config.lambda_scale
        
        # 2. Extract strictly the macro profile needed for the baselines
        macro_step_sec = config.lambda_scale * config.dt
        ds_macro = downsample_block_mean(raw_t, raw_pd, macro_step_sec, align='t0')
        
        # Truncate the macro demand to exactly match the hybrid macro horizon
        test_target_pd_macro = ds_macro['Pd'][:macro_horizon]

        # --- 2. Controller Setup ---
        if approach['strategy'] == 'SDP':
            # Fit Macro DTMC (300s) for outer loop
            ds_train_macro = downsample_block_mean(train_data['t'], train_data['Pd'], config.Ts, align='t0')
            mc_macro = fit_dtmc(ds_train_macro['Pd'], config.n_states, config.alpha)
            
            if approach.get('is_hybrid', False):
                # Fit Micro DTMC (5s) for inner simulation sweeps
                ds_train_micro = downsample_block_mean(train_data['t'], train_data['Pd'], config.dt, align='t0')
                
                # CRITICAL: Force the micro DTMC to use the exact same state boundaries
                mc_micro = fit_dtmc(ds_train_micro['Pd'], config.n_states, config.alpha, precomputed_edges=mc_macro['edges'])
                
                # Instantiate the Solver
                solver = HybridSDPSolver(config, mc_macro, mc_micro, variant=approach.get('sdp_variant', 'MEAN_PROXY'))
                
                # [THE MISSING PIECE]: Actually compute the tensors and instantiate the Hybrid Controller!
                policy_n, policy_pfc = solver.compute_policy_tensors(macro_horizon)
                
                # Assuming your HybridStochasticControl __init__ expects the grids and the dual policies
                controller = HybridStochasticControl(
                    config=config,
                    p_vals=mc_macro['levels'],
                    policy_n=policy_n,
                    policy_pfc=policy_pfc
                )
            else:
                # Wrap the naive SDP
                solver = BaselineSDPSolver(config, mc_macro)
                policy = solver.compute_policy_matrix(macro_horizon)
                base_ctrl = StochasticControl(mc_macro['levels'], config.n_vals, policy)
                controller = NaiveHybridWrapper(base_ctrl, test_target_pd_macro, config.n0, config.p_star)
                
        elif approach['strategy'] == 'HEURISTIC':
            if approach.get('is_hybrid', False):
                controller = HybridThresholdControl(config, macro_horizon, tau_relax=5.0)
            else:
                # Wrap the naive Threshold logic
                base_ctrl = ThresholdControl(config)
                controller = NaiveHybridWrapper(base_ctrl, test_target_pd_macro, config.n0, config.p_star)

        # --- 3. Unified Simulator Execution ---
        sim = HybridSimulator(config, test_target_pd, plant)
        sim.t_micro = sim_t  
        sim.raw_t = raw_t
        sim.raw_pd = raw_pd

        sim.run(controller)
        compute_time = time.time() - start_time
        
        # --- 4. Metric Extraction (Unified) ---
        op_cost = np.sum(sim.C_o_vec)
        sw_cost = np.sum(sim.C_s_vec)
        bat_cost = np.sum(sim.C_bat_vec)
        final_soc = sim.soc_history[-1]

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
    
class NaiveHybridWrapper:
    """
    Wraps the baseline MATLAB controllers to run inside the Multi-Timescale Hybrid physical world.
    """
    def __init__(self, base_controller, macro_demand_profile: np.ndarray, n0: int, p_star: float):
        self.base_controller = base_controller
        self.macro_demand = macro_demand_profile
        self.p_star = p_star
        
        # Pre-compute the entire deterministic sequence of module decisions
        self.n_decisions = base_controller.compute(macro_demand_profile, n0)

    def get_action(self, macro_step_k: int, current_pd: float, n_prev: int, current_soc: float) -> tuple[int, float]:
        # 1. Safely index the module count 
        idx = min(macro_step_k, len(self.n_decisions) - 1)
        n_k = self.n_decisions[idx]
        
        # 2. The baseline controller operates blindly to SoC. 
        # It attempts to supply the mean macro demand directly from the Fuel Cells.
        idx_demand = min(macro_step_k, len(self.macro_demand) - 1)
        pfc_requested = self.macro_demand[idx_demand]
        
        # 3. Strict physical clipping (FC cannot output negative power or exceed active capacity)
        pfc_k = np.clip(pfc_requested, 0.0, n_k * self.p_star)
        
        return n_k, pfc_k