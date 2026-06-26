# python/src/utils/evaluation.py
import time
import numpy as np
import pandas as pd
from src.utils.data_processing import downsample_block_mean, fit_dtmc
from src.simulator import Simulator

class VoyageBenchmarker:
    """
    Automated evaluation engine for Cross-Validation and Benchmarking.
    Handles data isolation, model training, and metric extraction.
    """
    def __init__(self, fleet_cache: dict, config, exclude_days: list = None):
        self.fleet_cache = fleet_cache
        self.config = config
        self.exclude_days = exclude_days or []
        # Sort all valid days to ensure chronological integrity
        self.valid_days = sorted([d for d in fleet_cache.keys() if d not in self.exclude_days])
        
    def _prepare_data(self, train_days: list, test_day: int):
        """Concatenates training data, fits the Markov Chain, and extracts the test target."""
        train_t, train_pd = [], []
        
        for d in train_days:
            data = self.fleet_cache[d]
            t_off = train_t[-1][-1] if train_t else 0.0
            train_t.append(data['t'] + t_off)
            train_pd.append(data['Pd'])
            
        t_concat = np.concatenate(train_t)
        pd_concat = np.concatenate(train_pd)
        
        # Train Macro DTMC using the corrected namespace 'alpha_mc'
        ds_train_macro = downsample_block_mean(t_concat, pd_concat, self.config.Ts, align='t0')
        mc_macro = fit_dtmc(ds_train_macro['Pd'], self.config.n_states, self.config.alpha_mc)
        
        # Prepare Test Data
        test_data = self.fleet_cache[test_day]
        ds_test = downsample_block_mean(test_data['t'], test_data['Pd'], self.config.Ts, align='t0')
        horizon_length = len(ds_test['Pd'])
        
        return mc_macro, test_data['Pd'], horizon_length

    def _evaluate_single_run(self, approach_factory, mc_model, test_pd, horizon_length):
        """
        Instantiates a completely fresh physical plant and controller to prevent 
        historical states (like transient penalties) from bleeding across runs.
        Tracks computation time for the offline solution + online simulation.
        """
        # Start the timing clock (perf_counter is the most accurate for benchmarking)
        start_time = time.perf_counter()
        
        # The factory yields a fresh controller (and computes the Bellman policy if SDP) and a fresh plant
        controller, plant = approach_factory(self.config, mc_model, horizon_length)
        
        sim = Simulator(self.config, test_pd, plant)
        sim.run(controller)
        
        # Stop the timing clock
        end_time = time.perf_counter()
        calc_time = end_time - start_time
        
        return {
            'Total Cost [€]': sum(sim.history.get('cost_total', [0.0])),
            'Operating Cost [€]': sum(sim.history.get('cost_o', [0.0])),
            'Switching Cost [€]': sum(sim.history.get('cost_s', [0.0])),
            'Transient Cost [€]': sum(sim.history.get('cost_trans', [0.0])),
            'Battery Cost [€]': sum(sim.history.get('cost_bat', [0.0])),
            'Computation Time [s]': calc_time
        }

    def compare_approaches(self, approaches: dict, train_days: list, test_day: int) -> pd.DataFrame:
        """Evaluates multiple control strategies against a hand-picked test day."""
        mc_model, test_pd, horizon_length = self._prepare_data(train_days, test_day)
        results = {}
        for name, factory in approaches.items():
            results[name] = self._evaluate_single_run(factory, mc_model, test_pd, horizon_length)
        return pd.DataFrame(results).T

    def run_leave_one_out(self, approach_factory) -> pd.DataFrame:
        """Leave-One-Out Cross-Validation: Tests on D, trains on all others."""
        results = {}
        for test_day in self.valid_days:
            train_days = [d for d in self.valid_days if d != test_day]
            mc_model, test_pd, horizon_length = self._prepare_data(train_days, test_day)
            results[f"Day {test_day}"] = self._evaluate_single_run(approach_factory, mc_model, test_pd, horizon_length)
            
        df = pd.DataFrame(results).T
        df.loc['Average'] = df.mean()
        return df

    def run_forward_chaining(self, approach_factory, min_train_days: int = 1) -> pd.DataFrame:
        """Chronological Forward Chaining: Tests time-series robustness."""
        results = {}
        for i in range(min_train_days, len(self.valid_days)):
            train_days = self.valid_days[:i]
            test_day = self.valid_days[i]
            mc_model, test_pd, horizon_length = self._prepare_data(train_days, test_day)
            results[f"Train 1-{train_days[-1]} -> Test {test_day}"] = self._evaluate_single_run(approach_factory, mc_model, test_pd, horizon_length)
            
        df = pd.DataFrame(results).T
        df.loc['Average'] = df.mean()
        return df