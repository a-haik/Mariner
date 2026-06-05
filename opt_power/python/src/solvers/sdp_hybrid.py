# python/src/solvers/sdp_hybrid.py
import numpy as np
from numba import njit
from config import HybridSimConfig
from src.plants.hybrid_plant import _get_pfc_bounds, _simulate_micro_physics

@njit(cache=True)
def _linear_interp_1d(x: float, grid: np.ndarray, values: np.ndarray) -> float:
    """Ultra-fast 1D linear interpolation."""
    if x <= grid[0]:
        return values[0]
    if x >= grid[-1]:
         return values[-1]
    idx = np.searchsorted(grid, x) - 1
    weight = (x - grid[idx]) / (grid[idx + 1] - grid[idx])
    return values[idx] + weight * (values[idx + 1] - values[idx])

# =============================================================================
# 1. EXACT_TREE: SCENARIO TREE PROPAGATION (The Ground Truth)
# =============================================================================
@njit
def _dfs_exact_tree(depth: int, lambda_scale: int, p_idx: int, soc_curr: float, 
                    n_k: int, n_prev: int, p_fc: float, path_prob: float, 
                    p_vals: np.ndarray, transition_matrix: np.ndarray, 
                    dt: float, e_bat: float, c_bat_kwh: float, n_eol: int, 
                    p_star: float, k_s: float, penalty_wall: float, 
                    V_next: np.ndarray, soc_vals: np.ndarray) -> float:
    """Recursively explores every possible Markov path over the lambda window."""
    p_d = p_vals[p_idx]
    p_bat_t = p_d - p_fc
    delta_soc = - (p_bat_t * (dt / 3600.0) / e_bat) * 100.0
    soc_next = soc_curr + delta_soc
    
    if soc_next < 0.0 or soc_next > 100.0:
        return path_prob * penalty_wall
        
    c_bat_step = (abs(p_bat_t) * (dt / 3600.0) * e_bat * c_bat_kwh) / (2.0 * n_eol * e_bat)
    
    if depth == lambda_scale - 1:
        exp_future = 0.0
        for i_next in range(len(p_vals)):
            prob = transition_matrix[p_idx, i_next]
            if prob > 0:
                val = _linear_interp_1d(soc_next, soc_vals, V_next[i_next, :])
                exp_future += prob * val
        return path_prob * (c_bat_step + exp_future)

    expected_branch_cost = 0.0
    for i_next in range(len(p_vals)):
        trans_prob = transition_matrix[p_idx, i_next]
        if trans_prob > 0:
            expected_branch_cost += _dfs_exact_tree(
                depth + 1, lambda_scale, i_next, soc_next, n_k, n_prev, p_fc, 
                path_prob * trans_prob, p_vals, transition_matrix, dt, e_bat, 
                c_bat_kwh, n_eol, p_star, k_s, penalty_wall, V_next, soc_vals
            )
            
    return path_prob * c_bat_step + expected_branch_cost

@njit(cache=True)
def _solve_exact_tree_bellman(T: int, p_vals: np.ndarray, n_vals: np.ndarray, 
                              soc_vals: np.ndarray, pfc_vals: np.ndarray, transition_matrix: np.ndarray, 
                              dt: float, lambda_scale: int, e_bat: float, c_bat_kwh: float, n_eol: int, 
                              p_star: float, k_s: float, penalty_wall: float, soc_terminal_target: float):
    """Wrapper that sweeps the Bellman equation using DFS for the expectation operator."""
    p_size, n_size, soc_size = len(p_vals), len(n_vals), len(soc_vals)
    V = np.full((T, p_size, n_size, soc_size), np.inf, dtype=np.float64)
    policy_n, policy_pfc = np.zeros((T, p_size, n_size, soc_size), dtype=np.int32), np.zeros((T, p_size, n_size, soc_size), dtype=np.float64)
    
    for i in range(p_size):
        for j in range(n_size):
            for s in range(soc_size):
                # 1. Evaluate Terminal Cost
                V[T - 1, i, j, s] = 0.0 if soc_vals[s] >= soc_terminal_target else penalty_wall
                
                # 2. Prevent Zero-Division: Maintain the current number of active modules
                policy_n[T - 1, i, j, s] = n_vals[j]
                
                # 3. Prevent bounds violations: Try to cover the mean demand safely
                safe_pfc = min(max(p_vals[i], 0.0), n_vals[j] * p_star)
                policy_pfc[T - 1, i, j, s] = safe_pfc
                    
    for t in range(T - 2, -1, -1):
        for i in range(p_size):
            for j in range(n_size):
                n_curr = n_vals[j]
                for s in range(soc_size):
                    soc_curr = soc_vals[s]
                    pfc_min, pfc_max = _get_pfc_bounds(soc_curr, p_vals[i], e_bat, dt)
                    best_cost, best_n_next, best_pfc = np.inf, n_curr, 0.0
                    
                    for a_idx in range(n_size):
                        n_next = n_vals[a_idx]
                        c_s = k_s * abs(n_next - n_curr)
                        c_o = (((pfc_vals / p_star) - n_next) ** 2) / n_next * lambda_scale
                        
                        for pfc_idx in range(len(pfc_vals)):
                            p_fc = pfc_vals[pfc_idx]
                            if p_fc < pfc_min or p_fc > pfc_max or p_fc > (n_next * p_star):
                                continue
                            
                            exp_cost_and_future = _dfs_exact_tree(
                                0, lambda_scale, i, soc_curr, n_next, n_curr, p_fc, 1.0,
                                p_vals, transition_matrix, dt, e_bat, c_bat_kwh, n_eol, 
                                p_star, k_s, penalty_wall, V[t + 1, :, a_idx, :], soc_vals
                            )
                            total_cost = c_s + c_o[pfc_idx] + exp_cost_and_future
                            if total_cost < best_cost:
                                best_cost, best_n_next, best_pfc = total_cost, n_next, p_fc
                                
                    V[t, i, j, s], policy_n[t, i, j, s], policy_pfc[t, i, j, s] = best_cost, best_n_next, best_pfc
    return policy_n, policy_pfc

# =============================================================================
# 2. MEAN_PROXY: DETERMINISTIC EXPECTED PATH (The Phase 3 Draft)
# =============================================================================
@njit(cache=True)
def _solve_mean_proxy_bellman(T: int, p_vals: np.ndarray, n_vals: np.ndarray, 
                              soc_vals: np.ndarray, pfc_vals: np.ndarray, transition_matrix: np.ndarray, 
                              dt: float, lambda_scale: int, e_bat: float, c_bat_kwh: float, n_eol: int, 
                              p_star: float, k_s: float, penalty_wall: float, soc_terminal_target: float):
    """The fast deterministic smoothing variant defined previously."""
    p_size, n_size, soc_size = len(p_vals), len(n_vals), len(soc_vals)
    V = np.full((T, p_size, n_size, soc_size), np.inf, dtype=np.float64)
    policy_n, policy_pfc = np.zeros((T, p_size, n_size, soc_size), dtype=np.int32), np.zeros((T, p_size, n_size, soc_size), dtype=np.float64)
    
    for i in range(p_size):
        for j in range(n_size):
            for s in range(soc_size):
                # 1. Evaluate Terminal Cost
                V[T - 1, i, j, s] = 0.0 if soc_vals[s] >= soc_terminal_target else penalty_wall
                
                # 2. Prevent Zero-Division: Maintain the current number of active modules
                policy_n[T - 1, i, j, s] = n_vals[j]
                
                # 3. Prevent bounds violations: Try to cover the mean demand safely
                safe_pfc = min(max(p_vals[i], 0.0), n_vals[j] * p_star)
                policy_pfc[T - 1, i, j, s] = safe_pfc
                    
    for t in range(T - 2, -1, -1):
        for i in range(p_size):
            p_d_micro = np.full(lambda_scale, p_vals[i], dtype=np.float64)
            for j in range(n_size):
                n_curr = n_vals[j]
                for s in range(soc_size):
                    soc_curr = soc_vals[s]
                    pfc_min, pfc_max = _get_pfc_bounds(soc_curr, p_vals[i], e_bat, dt)
                    best_cost, best_n_next, best_pfc = np.inf, n_curr, 0.0
                    
                    for a_idx in range(n_size):
                        n_next = n_vals[a_idx]
                        for pfc_idx in range(len(pfc_vals)):
                            p_fc = pfc_vals[pfc_idx]
                            if p_fc < pfc_min or p_fc > pfc_max or p_fc > (n_next * p_star):
                                continue
                            
                            step_cost, next_soc = _simulate_micro_physics(
                                soc_curr, n_next, n_curr, p_fc, p_d_micro, 
                                dt, e_bat, c_bat_kwh, n_eol, p_star, k_s, penalty_wall
                            )
                            if step_cost >= penalty_wall:
                                continue
                                
                            exp_future = 0.0
                            for i_next in range(p_size):
                                prob = transition_matrix[i, i_next]
                                if prob > 0:
                                    exp_future += prob * _linear_interp_1d(next_soc, soc_vals, V[t + 1, i_next, a_idx, :])
                                    
                            if step_cost + exp_future < best_cost:
                                best_cost, best_n_next, best_pfc = step_cost + exp_future, n_next, p_fc
                                
                    V[t, i, j, s], policy_n[t, i, j, s], policy_pfc[t, i, j, s] = best_cost, best_n_next, best_pfc
    return policy_n, policy_pfc

# =============================================================================
# 3. TENSOR_SWEEP: OFFLINE PRECOMPUTATION + ONLINE SWEEP
# =============================================================================
@njit(cache=True)
def _precompute_tensor_sweep_tensors(lambda_scale: int, mc_samples: int, p_vals: np.ndarray, 
                                     soc_vals: np.ndarray, n_vals: np.ndarray, pfc_vals: np.ndarray, 
                                     transition_matrix: np.ndarray, dt: float, e_bat: float, 
                                     c_bat_kwh: float, n_eol: int, p_star: float, k_s: float, 
                                     penalty_wall: float) -> tuple[np.ndarray, np.ndarray]:
    """Generates the offline expected transition and cost mapping via Monte Carlo."""
    p_size, n_size, soc_size, pfc_size = len(p_vals), len(n_vals), len(soc_vals), len(pfc_vals)
    exp_cost_tensor = np.full((p_size, n_size, pfc_size, soc_size), penalty_wall, dtype=np.float64)
    exp_soc_tensor = np.zeros((p_size, n_size, pfc_size, soc_size), dtype=np.float64)
    
    for i in range(p_size):
        for j in range(n_size):
            n_next = n_vals[j]
            for pfc_idx in range(pfc_size):
                for s in range(soc_size):
                    pfc_min, pfc_max = _get_pfc_bounds(soc_vals[s], p_vals[i], e_bat, dt)
                    if pfc_vals[pfc_idx] < pfc_min or pfc_vals[pfc_idx] > pfc_max or pfc_vals[pfc_idx] > (n_next * p_star):
                        continue
                        
                    sum_cost, sum_soc, valid_paths = 0.0, 0.0, 0
                    for _ in range(mc_samples):
                        p_d_micro = np.zeros(lambda_scale)
                        curr_p = i
                        p_d_micro[0] = p_vals[curr_p]
                        for t in range(1, lambda_scale):
                            r, cumprob = np.random.rand(), 0.0
                            for nxt in range(p_size):
                                cumprob += transition_matrix[curr_p, nxt]
                                if r <= cumprob:
                                    curr_p = nxt
                                    break
                            p_d_micro[t] = p_vals[curr_p]
                            
                        # Simulate WITHOUT switching cost (n_prev = n_next) because switching is state-dependent
                        cost, final_soc = _simulate_micro_physics(
                            soc_vals[s], n_next, n_next, pfc_vals[pfc_idx], 
                            p_d_micro, dt, e_bat, c_bat_kwh, n_eol, p_star, k_s, penalty_wall
                        )
                        if cost < penalty_wall:
                            sum_cost += cost
                            sum_soc += final_soc
                            valid_paths += 1
                            
                    if valid_paths > 0:
                        exp_cost_tensor[i, j, pfc_idx, s] = sum_cost / valid_paths
                        exp_soc_tensor[i, j, pfc_idx, s] = sum_soc / valid_paths
                        
    return exp_cost_tensor, exp_soc_tensor

@njit(cache=True)
def _solve_tensor_sweep_bellman(T: int, p_vals: np.ndarray, n_vals: np.ndarray, 
                                soc_vals: np.ndarray, pfc_vals: np.ndarray, transition_matrix: np.ndarray, 
                                exp_cost_tensor: np.ndarray, exp_soc_tensor: np.ndarray, 
                                k_s: float, p_star: float, penalty_wall: float, soc_terminal_target: float):
    """The lightning fast online matrix sweep."""
    p_size, n_size, soc_size = len(p_vals), len(n_vals), len(soc_vals)
    V = np.full((T, p_size, n_size, soc_size), np.inf, dtype=np.float64)
    policy_n, policy_pfc = np.zeros((T, p_size, n_size, soc_size), dtype=np.int32), np.zeros((T, p_size, n_size, soc_size), dtype=np.float64)
    
    for i in range(p_size):
        for j in range(n_size):
            for s in range(soc_size):
                # 1. Evaluate Terminal Cost
                V[T - 1, i, j, s] = 0.0 if soc_vals[s] >= soc_terminal_target else penalty_wall
                
                # 2. Prevent Zero-Division: Maintain the current number of active modules
                policy_n[T - 1, i, j, s] = n_vals[j]
                
                # 3. Prevent bounds violations: Try to cover the mean demand safely
                safe_pfc = min(max(p_vals[i], 0.0), n_vals[j] * p_star)
                policy_pfc[T - 1, i, j, s] = safe_pfc
                    
    for t in range(T - 2, -1, -1):
        for i in range(p_size):
            for j in range(n_size):
                n_curr = n_vals[j]
                for s in range(soc_size):
                    best_cost, best_n_next, best_pfc = np.inf, n_curr, 0.0
                    
                    for a_idx in range(n_size):
                        n_next = n_vals[a_idx]
                        c_s = k_s * abs(n_next - n_curr)
                        
                        for pfc_idx in range(len(pfc_vals)):
                            step_cost = exp_cost_tensor[i, a_idx, pfc_idx, s]
                            if step_cost >= penalty_wall:
                                continue
                            
                            next_soc = exp_soc_tensor[i, a_idx, pfc_idx, s]
                            exp_future = 0.0
                            for i_next in range(p_size):
                                prob = transition_matrix[i, i_next]
                                if prob > 0:
                                    exp_future += prob * _linear_interp_1d(next_soc, soc_vals, V[t + 1, i_next, a_idx, :])
                                    
                            total_cost = c_s + step_cost + exp_future
                            if total_cost < best_cost:
                                best_cost, best_n_next, best_pfc = total_cost, n_next, pfc_vals[pfc_idx]
                                
                    V[t, i, j, s], policy_n[t, i, j, s], policy_pfc[t, i, j, s] = best_cost, best_n_next, best_pfc
    return policy_n, policy_pfc

# =============================================================================
# FACADE INTERFACE
# =============================================================================
class HybridSDPSolver:
    """Facade routing the optimization to EXACT_TREE, MEAN_PROXY, or TENSOR_SWEEP."""
    def __init__(self, config: HybridSimConfig, mc_model: dict, variant: str = 'MEAN_PROXY'):
        self.config = config
        self.mc_model = mc_model
        self.variant = variant.upper()
        
        self.soc_grid = np.arange(0.0, 100.0 + self.config.soc_step, self.config.soc_step)
        max_power = np.max(self.config.n_vals) * self.config.p_star
        self.pfc_grid = np.arange(0.0, max_power + self.config.p_fc_step, self.config.p_fc_step)
        
        if self.variant not in ['EXACT_TREE', 'MEAN_PROXY', 'TENSOR_SWEEP']:
            raise ValueError("Solver variant must be 'EXACT_TREE', 'MEAN_PROXY', or 'TENSOR_SWEEP'.")

    def compute_policy_tensors(self, macro_horizon_length: int):
        print(f" -> Launching {self.variant} Solver...")
        
        if self.variant == 'EXACT_TREE':
            if self.config.lambda_scale > 5:
                print(" [!] WARNING: EXACT_TREE with lambda > 5 will cause combinatorial explosion.")
            return _solve_exact_tree_bellman(
                macro_horizon_length, self.mc_model['levels'], self.config.n_vals,
                self.soc_grid, self.pfc_grid, self.mc_model['P'], self.config.dt, 
                self.config.lambda_scale, self.config.e_bat, self.config.c_bat_kwh, 
                self.config.n_eol_cycles, self.config.p_star, self.config.k_s, 
                self.config.penalty_wall, self.config.soc_terminal_target
            )
            
        elif self.variant == 'MEAN_PROXY':
            return _solve_mean_proxy_bellman(
                macro_horizon_length, self.mc_model['levels'], self.config.n_vals,
                self.soc_grid, self.pfc_grid, self.mc_model['P'], self.config.dt, 
                self.config.lambda_scale, self.config.e_bat, self.config.c_bat_kwh, 
                self.config.n_eol_cycles, self.config.p_star, self.config.k_s, 
                self.config.penalty_wall, self.config.soc_terminal_target
            )
            
        elif self.variant == 'TENSOR_SWEEP':
            print(f" -> Triggering Offline MC Tensor Pre-Computation ({self.config.mc_samples} paths)...")
            cost_tens, soc_tens = _precompute_tensor_sweep_tensors(
                self.config.lambda_scale, self.config.mc_samples, self.mc_model['levels'],
                self.soc_grid, self.config.n_vals, self.pfc_grid, self.mc_model['P'],
                self.config.dt, self.config.e_bat, self.config.c_bat_kwh, 
                self.config.n_eol_cycles, self.config.p_star, self.config.k_s, 
                self.config.penalty_wall
            )
            print(" -> Tensors Cached. Initiating O(M^2) Online Sweep...")
            return _solve_tensor_sweep_bellman(
                macro_horizon_length, self.mc_model['levels'], self.config.n_vals,
                self.soc_grid, self.pfc_grid, self.mc_model['P'],
                cost_tens, soc_tens, self.config.k_s, self.config.p_star, 
                self.config.penalty_wall, self.config.soc_terminal_target
            )