# python/src/utils/evaluation.py
import time
import numpy as np
import pandas as pd
from src.utils.data_processing import downsample_block_mean, fit_dtmc
from src.simulator import Simulator
from src.utils.vault import RunVault

def print_markdown_table(df: pd.DataFrame):
    """Custom Markdown table generator for clean copy-pasting."""
    index_name = df.index.name if df.index.name else "Strategy"
    
    # Determine optimal column widths
    col_widths = [max(len(str(index_name)), max([len(str(idx)) for idx in df.index]))]
    for col in df.columns:
        max_len = max(len(str(col)), max([len(f"{x:.2f}") if isinstance(x, (float, np.floating)) else len(str(x)) for x in df[col]]))
        col_widths.append(max_len)
        
    def format_row(vals):
        return "| " + " | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(vals)) + " |"
        
    headers = [index_name] + list(df.columns)
    print(format_row(headers))
    print("|-" + "-|-".join("-" * w for w in col_widths) + "-|")
    
    for index, row in df.iterrows():
        vals = [index] + [f"{x:.2f}" if isinstance(x, (float, np.floating)) else x for x in row]
        print(format_row(vals))

class BenchmarkReport:
    """Lazy-evaluation interface to prevent RAM bloat during Cross-Validation."""
    def __init__(self, summary_df: pd.DataFrame, run_hashes: dict, vault: RunVault):
        self.summary = summary_df
        self.run_hashes = run_hashes
        self.vault = vault

    def get_telemetry(self, run_identifier: str) -> pd.DataFrame:
        """Loads the heavy time-series data from Parquet on demand."""
        if run_identifier not in self.run_hashes:
            raise KeyError(f"Run '{run_identifier}' not found. Available runs: {list(self.run_hashes.keys())}")
        return self.vault.load_telemetry(self.run_hashes[run_identifier])

class VoyageBenchmarker:
    """
    Automated evaluation engine for Cross-Validation and Benchmarking.
    Handles data isolation, model training, and metric extraction.
    """
    def __init__(self, fleet_cache: dict, config, exclude_days: list = None):
        self.fleet_cache = fleet_cache
        self.config = config
        self.exclude_days = exclude_days or []
        self.valid_days = sorted([d for d in fleet_cache.keys() if d not in self.exclude_days])
        self.vault = RunVault()
        
    def _prepare_data(self, train_days: list, test_day: int):
        train_t, train_pd = [], []
        for d in train_days:
            data = self.fleet_cache[d]
            t_off = train_t[-1][-1] if train_t else 0.0
            train_t.append(data['t'] + t_off)
            train_pd.append(data['Pd'])
            
        t_concat = np.concatenate(train_t)
        pd_concat = np.concatenate(train_pd)
        
        ds_train_macro = downsample_block_mean(t_concat, pd_concat, self.config.Ts, align='t0')
        mc_macro = fit_dtmc(ds_train_macro['Pd'], self.config.N_Pd, self.config.alpha_mc)
        
        test_data = self.fleet_cache[test_day]
        ds_test = downsample_block_mean(test_data['t'], test_data['Pd'], self.config.Ts, align='t0')
        horizon_length = len(ds_test['Pd'])
        
        # RETURN BOTH 1Hz AND 300s TEST ARRAYS
        return mc_macro, test_data['Pd'], ds_test['Pd'], horizon_length

    def _evaluate_single_run(self, approach_factory, mc_model, test_pd_micro, test_pd_macro, horizon_length, train_days, test_day):
        # 1. Safely extract physics identity without triggering the factory
        ctrl_name = getattr(approach_factory, 'controller_name', 'UnknownController')
        plant_name = getattr(approach_factory, 'plant_name', 'UnknownPlant')
        is_macro = getattr(approach_factory, 'is_macro', False)

        # 2. Generate Deterministic Hash
        hash_id = self.vault.generate_hash(ctrl_name, plant_name, train_days, test_day, is_macro, self.config)

        # 3. Cache Check
        cached_metrics = self.vault.get_metrics(hash_id)
        if cached_metrics is not None:
            print(f" [Vault] Cache HIT: {ctrl_name} (Day {test_day}). Skipping simulation.")
            return cached_metrics, hash_id

        print(f" [Vault] Cache MISS: Executing {ctrl_name} (Day {test_day})...")
        start_time = time.perf_counter()
        
        # 4. Execute the factory (This runs the heavy SDP math if applicable)
        controller, plant, is_macro_returned = approach_factory(self.config, mc_model, horizon_length)
        
        if is_macro_returned:
            # Snap the macro data strictly to the Markov grid
            levels = mc_model['levels']
            from src.utils.math_utils import nearest_index_1d
            test_pd = np.array([levels[nearest_index_1d(levels, val)] for val in test_pd_macro])
            dt_override = float(self.config.Ts)
        else:
            test_pd = test_pd_micro
            dt_override = None
            
        sim = Simulator(self.config, test_pd, plant, dt_override=dt_override)
        sim.run(controller)
        
        calc_time = time.perf_counter() - start_time
        
        metrics = {
            'Total Cost [€]': sum(sim.history.get('cost_total', [0.0])),
            'Operating Cost [€]': sum(sim.history.get('cost_o', [0.0])),
            'Switching Cost [€]': sum(sim.history.get('cost_s', [0.0])),
            'Transient Cost [€]': sum(sim.history.get('cost_trans', [0.0])),
            'Battery Cost [€]': sum(sim.history.get('cost_bat', [0.0])),
            'Computation Time [s]': calc_time
        }

        # 5. Save to Disk
        telemetry_df = pd.DataFrame(sim.history)
        self.vault.save_run(hash_id, ctrl_name, plant_name, train_days, test_day, is_macro_returned, metrics, telemetry_df)
        
        return metrics, hash_id

    def compare_approaches(self, approaches: dict, train_days: list, test_day: int) -> BenchmarkReport:
        mc_model, test_pd_micro, test_pd_macro, horizon = self._prepare_data(train_days, test_day)
        results, run_hashes = {}, {}
        for name, factory in approaches.items():
            metrics, hash_id = self._evaluate_single_run(factory, mc_model, test_pd_micro, test_pd_macro, horizon, train_days, test_day)
            results[name] = metrics
            run_hashes[name] = hash_id
        return BenchmarkReport(pd.DataFrame(results).T, run_hashes, self.vault)

    def run_leave_one_out(self, approach_factory) -> BenchmarkReport:
        results, run_hashes = {}, {}
        for test_day in self.valid_days:
            train_days = [d for d in self.valid_days if d != test_day]
            mc_model, test_pd_micro, test_pd_macro, horizon = self._prepare_data(train_days, test_day)
            
            run_id = f"Day {test_day}"
            metrics, hash_id = self._evaluate_single_run(approach_factory, mc_model, test_pd_micro, test_pd_macro, horizon, train_days, test_day)
            results[run_id] = metrics
            run_hashes[run_id] = hash_id
            
        df = pd.DataFrame(results).T
        df.loc['Average'] = df.mean()
        return BenchmarkReport(df, run_hashes, self.vault)

    def run_forward_chaining(self, approach_factory, min_train_days: int = 1) -> BenchmarkReport:
        results, run_hashes = {}, {}
        for i in range(min_train_days, len(self.valid_days)):
            train_days = self.valid_days[:i]
            test_day = self.valid_days[i]
            mc_model, test_pd_micro, test_pd_macro, horizon = self._prepare_data(train_days, test_day)
            
            run_id = f"Train 1-{train_days[-1]} -> Test {test_day}"
            metrics, hash_id = self._evaluate_single_run(approach_factory, mc_model, test_pd_micro, test_pd_macro, horizon, train_days, test_day)
            results[run_id] = metrics
            run_hashes[run_id] = hash_id
            
        df = pd.DataFrame(results).T
        df.loc['Average'] = df.mean()
        return BenchmarkReport(df, run_hashes, self.vault)