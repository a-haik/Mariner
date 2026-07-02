# python/src/controllers/augmented_hybrid_controllers.py
import numpy as np
from numba import njit
from src.controllers.base import ControlLaw
from src.core import State, Action
from src.config import SimConfig
from src.utils.math_utils import (
    nearest_index_1d, 
    bilinear_interp_2d,
    calc_cost_operational,
    calc_cost_switching,
    calc_cost_battery,
    calc_cost_transient
)

class AugmentedFCLockedControl(ControlLaw):
    """
    Simulates real-world hardware limits (PLCs). Snaps continuous reality to 
    the 4D discrete SDP grid, extracting a rigid fuel cell setpoint.
    """
    def __init__(self, p_grid: np.ndarray, n_vals: np.ndarray, soc_vals: np.ndarray, 
                 pfc_vals: np.ndarray, policy_n: np.ndarray, policy_pbatt: np.ndarray):
        self.p_grid = p_grid
        self.n_vals = n_vals
        self.soc_vals = soc_vals
        self.pfc_vals = pfc_vals
        self.policy_n = policy_n
        self.policy_pbatt = policy_pbatt
        self.current_step = 0

    def get_action(self, state: State) -> Action:
        t_idx = min(self.current_step, len(self.policy_n) - 1)
        
        # 1. Snap 4D continuous sensors to discrete grid
        idx_p = nearest_index_1d(self.p_grid, state.P_d)
        idx_n = nearest_index_1d(self.n_vals, float(state.n_prev))
        idx_soc = nearest_index_1d(self.soc_vals, state.soc)
        idx_pfc = nearest_index_1d(self.pfc_vals, state.p_fc_prev)
        
        # 2. Extract discrete intents from 4D array
        best_n_idx = self.policy_n[t_idx, idx_p, idx_n, idx_soc, idx_pfc]
        n_opt = int(self.n_vals[best_n_idx])
        p_batt_disc = float(self.policy_pbatt[t_idx, idx_p, idx_n, idx_soc, idx_pfc])
        
        # 3. Calculate Intended Rigid FC Setpoint based on the grid's belief
        discrete_p_d = self.p_grid[idx_p]
        p_fc_locked = discrete_p_d - p_batt_disc
        
        self.current_step += 1
        return Action(n_modules=n_opt, p_batt=p_batt_disc, p_fc=p_fc_locked)


class AugmentedPolicyControl(ControlLaw):
    """
    Smooths out the discrete policy matrices via 2D Bilinear Interpolation.
    Interpolates over the two continuous variables: SoC and P_fc_prev.
    """
    def __init__(self, p_grid: np.ndarray, n_vals: np.ndarray, soc_vals: np.ndarray, 
                 pfc_vals: np.ndarray, policy_n: np.ndarray, policy_pbatt: np.ndarray):
        self.p_grid = p_grid
        self.n_vals = n_vals
        self.soc_vals = soc_vals
        self.pfc_vals = pfc_vals
        self.policy_n = policy_n
        self.policy_pbatt = policy_pbatt
        self.current_step = 0

    def get_action(self, state: State) -> Action:
        t_idx = min(self.current_step, len(self.policy_n) - 1)
        
        # Snap parameters that act as "conditions" (Demand and Previous Modules)
        idx_p = nearest_index_1d(self.p_grid, state.P_d)
        idx_n = nearest_index_1d(self.n_vals, float(state.n_prev))
        
        # We must snap the continuous variables just to find the optimal module count 
        # (since module count is a discrete integer and cannot be interpolated)
        idx_soc = nearest_index_1d(self.soc_vals, state.soc)
        idx_pfc = nearest_index_1d(self.pfc_vals, state.p_fc_prev)
        
        best_n_idx = self.policy_n[t_idx, idx_p, idx_n, idx_soc, idx_pfc]
        n_opt = int(self.n_vals[best_n_idx])
        
        # 2D Interpolation over continuous SoC and P_fc_prev dimensions
        # We slice the matrix to lock in the Demand and the n_prev.
        slice_2d = self.policy_pbatt[t_idx, idx_p, idx_n, :, :] 
        p_batt_opt = bilinear_interp_2d(self.soc_vals, self.pfc_vals, slice_2d, state.soc, state.p_fc_prev)
        
        p_fc_locked = state.P_d - p_batt_opt
        
        self.current_step += 1
        return Action(n_modules=n_opt, p_batt=p_batt_opt, p_fc=p_fc_locked)


@njit(cache=True)
def _run_augmented_1_step_lookahead(P_d_real: float, soc_real: float, n_prev: int, pfc_prev: float, Ts: float, 
                                    p_max: float, p_nom: float, k_fc: float, k_h2: float, S_max: float, 
                                    tau_fc: float, alpha_fc: float, a0: float, a1: float, a2: float,
                                    C_bat: float, C_rep: float, E_life: float, lambda_trans: float, 
                                    soc_min: float, soc_max: float, pb_vals: np.ndarray, n_vals: np.ndarray, 
                                    soc_vals: np.ndarray, pfc_vals: np.ndarray,
                                    transition_row: np.ndarray, V_next: np.ndarray):
    """JIT Engine evaluating true continuous 4D cost using the Unified Cost Engine."""
    
    n_size = len(n_vals)
    soc_size = len(soc_vals)
    pfc_size = len(pfc_vals)
    p_size = len(transition_row)

    # 1. Pre-collapse the Expected Value over the stochastic Demand dimension
    exp_v = np.zeros((n_size, soc_size, pfc_size), dtype=np.float64)
    for a_idx in range(n_size):
        for k_idx in range(soc_size):
            for l_idx in range(pfc_size):
                s = 0.0
                for i_next in range(p_size):
                    s += transition_row[i_next] * V_next[i_next, a_idx, k_idx, l_idx]
                exp_v[a_idx, k_idx, l_idx] = s

    best_cost = np.inf
    best_n = n_vals[0]
    best_pbatt = 0.0

    # 2. Grid search over action space using exact continuous physical reality
    for a_idx in range(n_size):
        n_curr = n_vals[a_idx]
        for pbatt in pb_vals:
            # Kinematics
            soc_next = soc_real - (pbatt * (Ts / 3600.0)) / C_bat
            if soc_next < soc_min or soc_next > soc_max:
                continue

            p_fc_curr = P_d_real - pbatt
            
            # Constraints
            if p_fc_curr < 0:
                continue
            if n_curr > 0 and (p_fc_curr / n_curr) > p_max:
                continue
            if n_curr == 0 and p_fc_curr > 0:
                continue

            # Centralized Cost Engine
            c_o = calc_cost_operational(n_curr, p_fc_curr, p_nom, tau_fc, alpha_fc, k_fc, k_h2, a0, a1, a2, Ts)
            c_s = calc_cost_switching(n_curr, n_prev, k_fc, S_max)
            c_bat = calc_cost_battery(pbatt, Ts, C_rep, E_life)
            c_trans = calc_cost_transient(n_curr, n_prev, p_fc_curr, pfc_prev, lambda_trans)

            # 2D Interpolation over expected future surface
            z_mat = exp_v[a_idx, :, :]
            exp_future = bilinear_interp_2d(soc_vals, pfc_vals, z_mat, soc_next, p_fc_curr)

            total_cost = c_o + c_s + c_bat + c_trans + exp_future
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_n = n_curr
                best_pbatt = pbatt

    return best_n, best_pbatt

class AugmentedValueControl(ControlLaw):
    """
    Bypasses the Policy Matrix entirely. Solves the 4D Bellman equation online for the 
    exact continuous state, using the Value Matrix only for future expectations.
    """
    def __init__(self, config: SimConfig, p_grid: np.ndarray, transition_matrix: np.ndarray, 
                 V_matrix: np.ndarray):
        self.config = config
        self.p_grid = p_grid
        self.transition_matrix = transition_matrix
        self.V = V_matrix
        self.current_step = 0

    def get_action(self, state: State) -> Action:
        t_idx = min(self.current_step, len(self.V) - 1)
        
        if t_idx >= len(self.V) - 1:
            V_next = np.zeros((len(self.p_grid), len(self.config.n_vals), len(self.config.soc_vals), len(self.config.pfc_vals)))
        else:
            V_next = self.V[t_idx + 1]

        idx_p = nearest_index_1d(self.p_grid, state.P_d)
        transition_row = self.transition_matrix[idx_p, :]

        best_n, best_pbatt = _run_augmented_1_step_lookahead(
            P_d_real=state.P_d, soc_real=state.soc, n_prev=state.n_prev, pfc_prev=state.p_fc_prev, Ts=float(self.config.Ts),
            p_max=float(self.config.p_max), p_nom=float(self.config.p_nom), k_fc=float(self.config.k_fc),
            k_h2=float(self.config.k_h2), S_max=float(self.config.S_max), tau_fc=float(self.config.tau_fc),
            alpha_fc=float(self.config.alpha_fc), a0=float(self.config.a0), a1=float(self.config.a1),
            a2=float(self.config.a2), C_bat=float(self.config.C_bat), C_rep=float(self.config.C_rep),
            E_life=float(self.config.E_life), lambda_trans=float(self.config.lambda_trans),
            soc_min=float(self.config.soc_min), soc_max=float(self.config.soc_max),
            pb_vals=self.config.pb_vals, n_vals=self.config.n_vals, soc_vals=self.config.soc_vals, 
            pfc_vals=self.config.pfc_vals, transition_row=transition_row, V_next=V_next
        )

        p_fc_locked = state.P_d - best_pbatt
        
        self.current_step += 1
        return Action(n_modules=int(best_n), p_batt=best_pbatt, p_fc=p_fc_locked)