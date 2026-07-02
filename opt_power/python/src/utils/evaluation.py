# python/src/utils/evaluation.py
import time
import numpy as np
import pandas as pd
from src.utils.data_processing import downsample_block_mean, fit_dtmc
from src.simulator import Simulator
from src.utils.vault import ModelVault

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
    """In-memory evaluation interface. Telemetry is no longer written to disk."""
    def __init__(self, summary_df: pd.DataFrame, telemetry_dict: dict):
        self.summary = summary_df
        self.telemetry = telemetry_dict

    def get_telemetry(self, run_identifier: str) -> pd.DataFrame:
        """Instantly returns the DataFrame from RAM."""
        if run_identifier not in self.telemetry:
            raise KeyError(f"Run '{run_identifier}' not found.")
        return self.telemetry[run_identifier]

class VoyageBenchmarker:
    """
    Automated evaluation engine.
    Now separated into Phase 1 (Model Training/Fetching) and Phase 2 (Fast Simulation).
    """
    def __init__(self, fleet_cache: dict, config, exclude_days: list = None):
        self.fleet_cache = fleet_cache
        self.config = config
        self.exclude_days = exclude_days or []
        self.valid_days = sorted([d for d in fleet_cache.keys() if d not in self.exclude_days])
        
        # Initialize the new ModelVault pointing to the path in SimConfig
        self.vault = ModelVault(self.config.vault_dir)

    def _get_or_compute_models(self, train_days: list, solver_cls, horizon_length: int):
        total_offline_time = 0.0
        
        # --- 1. MARKOV MODEL ---
        markov_hash = self.vault.generate_markov_hash(train_days, self.config)
        loaded_mc = self.vault.load_markov_model(markov_hash)
        
        if loaded_mc is not None:
            mc_model, mc_time = loaded_mc
        else:
            print(f" [Vault] Cache MISS: Training Markov Chain for days {train_days}...")
            train_t, train_pd = [], []
            for d in train_days:
                data = self.fleet_cache[d]
                t_off = train_t[-1][-1] if train_t else 0.0
                train_t.append(data['t'] + t_off)
                train_pd.append(data['Pd'])
                
            t_concat = np.concatenate(train_t)
            pd_concat = np.concatenate(train_pd)
            
            ds_train = downsample_block_mean(t_concat, pd_concat, self.config.Ts, align='t0')
            
            # Start Markov Timer
            start_t = time.perf_counter()
            mc_model = fit_dtmc(ds_train['Pd'], self.config.N_Pd, self.config.alpha_mc)
            mc_time = time.perf_counter() - start_t
            
            mc_model['Delta'] = self.config.Ts  
            self.vault.save_markov_model(markov_hash, mc_model, train_days, offline_time=mc_time)
            
        total_offline_time += mc_time
        
        # --- 2. SDP BELLMAN MATRICES ---
        raw_solution = None
        if solver_cls is not None:
            sdp_hash = self.vault.generate_sdp_hash(markov_hash, solver_cls.__name__, horizon_length, self.config)
            loaded_sdp = self.vault.load_sdp_model(sdp_hash)
            
            if loaded_sdp is not None:
                raw_solution, sdp_time = loaded_sdp
            else:
                print(f" [Vault] Cache MISS: Solving SDP Bellman matrices ({solver_cls.__name__})...")
                solver = solver_cls(self.config, mc_model)
                
                # Start Bellman Timer
                start_t = time.perf_counter()
                raw_solution = solver.compute_solution(horizon_length)
                sdp_time = time.perf_counter() - start_t
                
                self.vault.save_sdp_model(sdp_hash, markov_hash, solver_cls.__name__, raw_solution, offline_time=sdp_time)
                
            total_offline_time += sdp_time
            
        return mc_model, raw_solution, total_offline_time

    def _run_simulation(self, approach_factory, mc_model, raw_solution, test_pd_micro, test_pd_macro, horizon_length, offline_time):
        """Phase 2: Bypasses training entirely and just executes the continuous environment."""
        is_macro_returned = approach_factory.is_macro
        
        # Pass the pre-computed math into the factory
        controller, plant, _ = approach_factory(self.config, mc_model, horizon_length, raw_solution=raw_solution)
        
        if is_macro_returned:
            levels = mc_model['levels']
            from src.utils.math_utils import nearest_index_1d
            test_pd = np.array([levels[nearest_index_1d(levels, val)] for val in test_pd_macro])
            dt_override = float(self.config.Ts)
        else:
            test_pd = test_pd_micro
            dt_override = None
            
        sim = Simulator(self.config, test_pd, plant, dt_override=dt_override)
        start_time = time.perf_counter()
        sim_results = sim.run(controller)
        calc_time = time.perf_counter() - start_time
        
        # Include offline_time in the final dictionary!
        metrics = {
            'Total Cost [€]': sim_results['total_cost'],
            'Operating Cost [€]': sum(sim.history.get('cost_o', [0.0])),
            'Switching Cost [€]': sum(sim.history.get('cost_s', [0.0])),
            'Battery Cost [€]': sum(sim.history.get('cost_bat', [0.0])),
            'Term. Switch Cost [€]': sim_results['terminal_n_cost'],
            'Term. SoC Cost [€]': sim_results['terminal_soc_cost'],
            'Offline Compute Time [s]': offline_time,
            'Online Compute Time [s]': calc_time        
        }
        
        telemetry_df = pd.DataFrame(sim.history)
        return metrics, telemetry_df

    def compare_approaches(self, approaches: dict, train_days: list, test_day: int) -> BenchmarkReport:
        test_data = self.fleet_cache[test_day]
        ds_test = downsample_block_mean(test_data['t'], test_data['Pd'], self.config.Ts, align='t0')
        horizon = len(ds_test['Pd'])
        
        results, telemetry_dict = {}, {}
        
        for name, factory in approaches.items():
            solver_cls = getattr(factory, 'solver_cls', None)
            
            mc_model, raw_solution, offline_time = self._get_or_compute_models(train_days, solver_cls, horizon)
            metrics, df = self._run_simulation(
                factory, mc_model, raw_solution, 
                test_data['Pd'], ds_test['Pd'], horizon, offline_time
            )
            
            results[name] = metrics
            telemetry_dict[name] = df
            
        return BenchmarkReport(pd.DataFrame(results).T, telemetry_dict)

    def run_leave_one_out(self, approach_factory) -> BenchmarkReport:
        results, telemetry_dict = {}, {}
        solver_cls = getattr(approach_factory, 'solver_cls', None)
        
        for test_day in self.valid_days:
            train_days = [d for d in self.valid_days if d != test_day]
            
            test_data = self.fleet_cache[test_day]
            ds_test = downsample_block_mean(test_data['t'], test_data['Pd'], self.config.Ts, align='t0')
            horizon = len(ds_test['Pd'])
            
            run_id = f"Day {test_day}"
            
            mc_model, raw_solution, offline_time = self._get_or_compute_models(train_days, solver_cls, horizon)
            metrics, df = self._run_simulation(
                approach_factory, mc_model, raw_solution, 
                test_data['Pd'], ds_test['Pd'], horizon, offline_time
            )
            
            results[run_id] = metrics
            telemetry_dict[run_id] = df
            
        df_results = pd.DataFrame(results).T
        df_results.loc['Average'] = df_results.mean()
        return BenchmarkReport(df_results, telemetry_dict)

    def run_forward_chaining(self, approach_factory, min_train_days: int = 1) -> BenchmarkReport:
        results, telemetry_dict = {}, {}
        solver_cls = getattr(approach_factory, 'solver_cls', None)
        
        for i in range(min_train_days, len(self.valid_days)):
            train_days = self.valid_days[:i]
            test_day = self.valid_days[i]
            
            test_data = self.fleet_cache[test_day]
            ds_test = downsample_block_mean(test_data['t'], test_data['Pd'], self.config.Ts, align='t0')
            horizon = len(ds_test['Pd'])
            
            run_id = f"Train {train_days[-1]} -> Test {test_day}"
            
            mc_model, raw_solution, offline_time = self._get_or_compute_models(train_days, solver_cls, horizon)
            metrics, df = self._run_simulation(
                approach_factory, mc_model, raw_solution, 
                test_data['Pd'], ds_test['Pd'], horizon, offline_time
            )
            
            results[run_id] = metrics
            telemetry_dict[run_id] = df
            
        df_results = pd.DataFrame(results).T
        df_results.loc['Average'] = df_results.mean()
        return BenchmarkReport(df_results, telemetry_dict)