import numpy as np
from numba import njit
from src.controllers.base import ControlLaw

# =============================================================================
# NUMBA CORE ROUTINES
# =============================================================================

@njit(cache=True)
def _compute_ech_discrete_core(P_d_trajectory: np.ndarray, p_vals: np.ndarray, n_vals: np.ndarray, 
                               n0: int, dt: float, k_s: float, cost_matrix: np.ndarray, 
                               P_hold: np.ndarray) -> np.ndarray:
    """ JIT-compiled engine for Variation 1: Expected Cost Full Discrete Search. """
    T_horizon = len(P_d_trajectory)
    n_control = np.zeros(T_horizon, dtype=np.int32)
    n_control[0] = n0
    
    n_prev_idx = np.abs(n_vals - n0).argmin()
    num_n_states = len(n_vals)
    
    for t in range(1, T_horizon):
        P_d_curr = P_d_trajectory[t]
        i = np.abs(p_vals - P_d_curr).argmin()
        time_left = (T_horizon - t - 1) * dt
        
        # We now minimize the RATE ($/sec), not the sum
        best_cost_rate = np.inf
        best_n_idx = n_prev_idx
        
        for j in range(num_n_states):
            p_h = P_hold[i, j]
            if p_h >= 0.9999:
                e_t_hold = time_left
            else:
                e_t_hold = min(dt / (1.0 - p_h), time_left)
                
            if e_t_hold <= 0:
                continue # Edge case safeguard for the very last step
                
            switch_cost = k_s * abs(n_vals[j] - n_vals[n_prev_idx])
            
            # AMORTIZED COST RATE ($/second)
            cost_rate_j = cost_matrix[j, i] + (switch_cost / e_t_hold)
            
            if cost_rate_j < best_cost_rate:
                best_cost_rate = cost_rate_j
                best_n_idx = j
                
        n_control[t] = n_vals[best_n_idx]
        n_prev_idx = best_n_idx
        
    return n_control


@njit(cache=True)
def _compute_ech_target_step_core(P_d_trajectory: np.ndarray, p_vals: np.ndarray, n_vals: np.ndarray, 
                                  n0: int, dt: float, k_s: float, cost_matrix: np.ndarray, 
                                  P_hold: np.ndarray, ideal_targets: np.ndarray) -> np.ndarray:
    """ JIT-compiled engine for Variation 2: Expected Cost Target-or-Step Hybrid. """
    T_horizon = len(P_d_trajectory)
    n_control = np.zeros(T_horizon, dtype=np.int32)
    n_control[0] = n0
    
    n_prev_idx = np.abs(n_vals - n0).argmin()
    
    for t in range(1, T_horizon):
        P_d_curr = P_d_trajectory[t]
        i = np.abs(p_vals - P_d_curr).argmin()
        time_left = (T_horizon - t - 1) * dt
        
        target_idx = ideal_targets[i]
        
        if target_idx == n_prev_idx:
            n_control[t] = n_vals[n_prev_idx]
            continue
            
        step_dir = 1 if target_idx > n_prev_idx else -1
        step_idx = n_prev_idx + step_dir
        
        candidates = np.array([target_idx, step_idx])
        unique_candidates = np.unique(candidates) 
        
        # AMORTIZED BASELINE: The cost rate of doing nothing is just its operating cost.
        best_cost_rate = cost_matrix[n_prev_idx, i]
        best_n_idx = n_prev_idx
        
        for j in unique_candidates:
            p_h = P_hold[i, j]
            if p_h >= 0.9999:
                e_t_hold = time_left
            else:
                e_t_hold = min(dt / (1.0 - p_h), time_left)
                
            if e_t_hold <= 0:
                continue
                
            switch_cost = k_s * abs(n_vals[j] - n_vals[n_prev_idx])
            
            # AMORTIZED COST RATE ($/second)
            cost_rate_j = cost_matrix[j, i] + (switch_cost / e_t_hold)
            
            if cost_rate_j < best_cost_rate:
                best_cost_rate = cost_rate_j
                best_n_idx = j
                
        n_control[t] = n_vals[best_n_idx]
        n_prev_idx = best_n_idx
        
    return n_control

# =============================================================================
# CONTROLLER CLASSES
# =============================================================================

class ExpectedCostHeuristicBase(ControlLaw):
    """ Shared precomputation logic for the Expected Cost heuristics using Top-M Comfort Zones. """
    def __init__(self, p_vals: np.ndarray, n_vals: np.ndarray, trans_mat_macro: np.ndarray, 
                 cost_matrix: np.ndarray, dt: float, k_s: float, tolerance: int = 1):
        self.p_vals = p_vals
        self.n_vals = n_vals
        self.dt = dt
        self.k_s = k_s
        self.cost_matrix = cost_matrix
        self.tolerance = tolerance 
        
        num_p = len(p_vals)
        num_n = len(n_vals)
        
        # 1. Precompute absolute ideal discrete targets (For Variation 2 Candidate Generation)
        self.ideal_targets = np.argmin(cost_matrix, axis=0) 
        
        # 2. Build Top-M Comfort Zones
        # K_zones[j] stores the demand indices 'i' where module count 'j' is in the Top M cheapest options
        K_zones = {j: [] for j in range(num_n)}
        for i in range(num_p):
            # Sort the column of costs for this demand state ascending
            costs_l = cost_matrix[:, i]
            top_m_indices = np.argsort(costs_l)[:self.tolerance]
            
            # Map this demand state to the comfort zones of those top M module counts
            for j in top_m_indices:
                K_zones[j].append(i)
                
        # 3. Calculate the P_hold matrix
        self.P_hold = np.zeros((num_p, num_n))
        for j in range(num_n):
            K_j_indices = np.array(K_zones[j], dtype=int)
            if len(K_j_indices) > 0:
                # Sum transition probabilities into the Top-M comfort zone for state j
                self.P_hold[:, j] = np.sum(trans_mat_macro[:, K_j_indices], axis=1)

class ECHDiscreteSearch(ExpectedCostHeuristicBase):
    """ Variation 1: Full Discrete Search """
    def compute(self, P_d: np.ndarray, n0: int) -> np.ndarray:
        return _compute_ech_discrete_core(
            P_d, self.p_vals, self.n_vals, n0, self.dt, self.k_s, 
            self.cost_matrix, self.P_hold
        )

class ECHTargetStep(ExpectedCostHeuristicBase):
    """ Variation 2: Analytical Target-or-Step Hybrid """
    def compute(self, P_d: np.ndarray, n0: int) -> np.ndarray:
        return _compute_ech_target_step_core(
            P_d, self.p_vals, self.n_vals, n0, self.dt, self.k_s, 
            self.cost_matrix, self.P_hold, self.ideal_targets
        )