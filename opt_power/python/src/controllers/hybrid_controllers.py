# python/src/controllers/hybrid_controllers.py
import numpy as np
from numba import njit
from src.controllers.base import ControlLaw
from src.core import State, Action
from src.config import SimConfig
from src.utils.math_utils import (
    nearest_index_1d, 
    bilinear_interp, 
    linear_interp_1d,
    calc_cost_operational,
    calc_cost_switching,
    calc_cost_battery,
    get_exact_index_1d,
    calculate_continuous_bounds,
    gss_standard
)

class HybridFCLockedControl(ControlLaw):
    """
    Simulates real-world hardware limits (PLCs). Snaps continuous reality to 
    the discrete SDP grid, extracting a rigid fuel cell setpoint. The battery 
    absorbs the high-frequency physical turbulence.
    """
    def __init__(self, p_grid: np.ndarray, n_vals: np.ndarray, soc_vals: np.ndarray, 
                 policy_n: np.ndarray, policy_pbatt: np.ndarray):
        self.p_grid = p_grid
        self.n_vals = n_vals
        self.soc_vals = soc_vals
        self.policy_n = policy_n
        self.policy_pbatt = policy_pbatt
        self.current_step = 0

    def get_action(self, state: State) -> Action:
        t_idx = min(self.current_step, len(self.policy_n) - 1)
        
        # 1. Snap continuous sensors to discrete grid
        idx_p = nearest_index_1d(self.p_grid, state.P_d)
        idx_n = nearest_index_1d(self.n_vals, float(state.n_prev))
        idx_soc = nearest_index_1d(self.soc_vals, state.soc)
        
        # 2. Extract discrete intents
        best_n_idx = self.policy_n[t_idx, idx_p, idx_n, idx_soc]
        n_opt = int(self.n_vals[best_n_idx])
        p_batt_disc = float(self.policy_pbatt[t_idx, idx_p, idx_n, idx_soc])
        
        # 3. Calculate Intended Rigid FC Setpoint
        discrete_p_d = self.p_grid[idx_p]
        p_fc_locked = discrete_p_d - p_batt_disc
        
        self.current_step += 1
        return Action(n_modules=n_opt, p_batt=p_batt_disc, p_fc=p_fc_locked)

class HybridPolicyControl(ControlLaw):
    """
    Attempts to smooth out the discrete policy matrices via 2D Bilinear Interpolation.
    Highlights the dangers of interpolating 'bang-bang' optimal edges.
    """
    def __init__(self, config: SimConfig, p_grid: np.ndarray, n_vals: np.ndarray, soc_vals: np.ndarray, 
                 policy_n: np.ndarray, policy_pbatt: np.ndarray, is_macro: bool = False):
        self.config = config
        self.p_grid = p_grid
        self.n_vals = n_vals
        self.soc_vals = soc_vals
        self.policy_n = policy_n
        self.policy_pbatt = policy_pbatt
        self.current_step = 0
        self.is_macro = is_macro

    def get_action(self, state: State) -> Action:
        t_idx = min(self.current_step, len(self.policy_n) - 1)
        
        # Modules are integers, so they must still be snapped
        idx_p = nearest_index_1d(self.p_grid, state.P_d)
        idx_n = nearest_index_1d(self.n_vals, float(state.n_prev))
        idx_soc = nearest_index_1d(self.soc_vals, state.soc)
        
        best_n_idx = self.policy_n[t_idx, idx_p, idx_n, idx_soc]
        n_opt = int(self.n_vals[best_n_idx])
        
        use_sg = getattr(self.config, 'use_smart_grid', False)
        if self.is_macro and use_sg:
            p_batt_opt = float(self.policy_pbatt[t_idx, idx_p, idx_n, idx_soc])
        else:
            slice_2d = self.policy_pbatt[t_idx, :, idx_n, :] 
            p_batt_opt = bilinear_interp(self.p_grid, self.soc_vals, slice_2d, state.P_d, state.soc)
        
        p_fc_locked = state.P_d - p_batt_opt
        
        self.current_step += 1
        return Action(n_modules=n_opt, p_batt=p_batt_opt, p_fc=p_fc_locked)

@njit(cache=True)
def _run_1_step_lookahead(P_d_real: float, soc_real: float, n_prev: int, Ts: float, 
                          p_max: float, p_nom: float, k_fc: float, k_h2: float, S_max: float, 
                          tau_fc: float, alpha_fc: float, a0: float, a1: float, a2: float,
                          Q_bat: float, C_rep: float, E_life: float, soc_min: float, soc_max: float,
                          pb_vals: np.ndarray, n_vals: np.ndarray, soc_vals: np.ndarray,
                          transition_row: np.ndarray, V_next: np.ndarray,
                          use_exact: bool, dSoC: float, is_macro: bool): # <-- ADDED is_macro
                          
    n_size = len(n_vals)
    soc_size = len(soc_vals)
    p_size = len(transition_row)

    exp_v = np.zeros((n_size, soc_size))
    for a_idx in range(n_size):
        for k_idx in range(soc_size):
            s = 0.0
            for i_next in range(p_size):
                s += transition_row[i_next] * V_next[i_next, a_idx, k_idx]
            exp_v[a_idx, k_idx] = s

    best_cost = np.inf
    best_n = n_vals[0]
    best_pbatt = 0.0
    
    pb_min_config = pb_vals[0]
    pb_max_config = pb_vals[-1]

    for a_idx in range(n_size):
        n_next = n_vals[a_idx]
        
        if is_macro:
            # ORIGINAL DISCRETE LOGIC
            for pbatt in pb_vals:
                epsilon = 1e-5
                soc_next = soc_real - (pbatt * (Ts / 3600.0)) / Q_bat
                p_fc = P_d_real - pbatt
                
                if p_fc < -epsilon:
                    continue
                if n_next > 0 and (p_fc / n_next) > p_max + epsilon:
                    continue
                if n_next == 0 and p_fc > epsilon:
                    continue

                C_o = calc_cost_operational(n_next, p_fc, p_nom, tau_fc, alpha_fc, k_fc, k_h2, a0, a1, a2, Ts)
                C_s = calc_cost_switching(n_next, n_prev, k_fc, S_max)
                C_bat = calc_cost_battery(pbatt, Ts, C_rep, E_life)

                if use_exact:
                    s_idx = get_exact_index_1d(soc_next, soc_min, dSoC, soc_size - 1)
                    exp_future = exp_v[a_idx, s_idx]
                else:
                    exp_future = linear_interp_1d(soc_vals, exp_v[a_idx, :], soc_next)

                total_cost = C_o + C_s + C_bat + exp_future
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_n = n_next
                    best_pbatt = pbatt
        else:
            # CONTINUOUS OPTIMIZER LOGIC
            pb_min, pb_max = calculate_continuous_bounds(P_d_real, soc_real, n_next, Ts, p_max, Q_bat, soc_min, soc_max, pb_min_config, pb_max_config)
            
            if pb_min > pb_max + 1e-5:
                continue # Bounds are physically impossible
                
            opt_pb, total_cost = gss_standard(pb_min, pb_max, 1.0, P_d_real, soc_real, n_next, n_prev, Ts, 
                                              p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, 
                                              Q_bat, C_rep, E_life, soc_vals, exp_v[a_idx, :])
            if total_cost < best_cost:
                best_cost = total_cost
                best_n = n_next
                best_pbatt = opt_pb

    return best_n, best_pbatt

class HybridValueControl(ControlLaw):
    """
    Bypasses the Policy Matrix entirely. Solves the Bellman equation online for the 
    exact continuous state (P_d, SoC), using the Value Matrix only for future expectations.
    """
    def __init__(self, config: SimConfig, p_grid: np.ndarray, transition_matrix: np.ndarray, 
                 V_matrix: np.ndarray, is_macro: bool = False):
        self.config = config
        self.p_grid = p_grid
        self.transition_matrix = transition_matrix
        self.V = V_matrix
        self.current_step = 0
        self.is_macro = is_macro

    def get_action(self, state: State) -> Action:
        t_idx = min(self.current_step, len(self.V) - 1)
        
        if t_idx >= len(self.V) - 1:
            V_next = np.zeros((len(self.p_grid), len(self.config.n_vals), len(self.config.soc_vals)))
        else:
            V_next = self.V[t_idx + 1]

        if self.is_macro:
            P_d_eval = self.p_grid[nearest_index_1d(self.p_grid, state.P_d)]
            n_eval = int(self.config.n_vals[nearest_index_1d(self.config.n_vals, float(state.n_prev))])
            soc_eval = self.config.soc_vals[nearest_index_1d(self.config.soc_vals, state.soc)]
        else:
            P_d_eval = state.P_d
            n_eval = state.n_prev
            soc_eval = state.soc

        idx_p = nearest_index_1d(self.p_grid, P_d_eval)
        transition_row = self.transition_matrix[idx_p, :]
        
        use_sg = getattr(self.config, 'use_smart_grid', False)
        use_exact = use_sg and self.is_macro
        dP = getattr(self.config, 'dP', 140.0)
        dSoC = (dP * float(self.config.Ts)) / (float(self.config.Q_bat) * 3600.0) if use_sg else 0.0

        best_n, best_pbatt = _run_1_step_lookahead(
            P_d_real=P_d_eval, soc_real=soc_eval, n_prev=n_eval, Ts=float(self.config.Ts),
            p_max=float(self.config.p_max), p_nom=float(self.config.p_nom), k_fc=float(self.config.k_fc),
            k_h2=float(self.config.k_h2), S_max=float(self.config.S_max), tau_fc=float(self.config.tau_fc),
            alpha_fc=float(self.config.alpha_fc), a0=float(self.config.a0), a1=float(self.config.a1),
            a2=float(self.config.a2), Q_bat=float(self.config.Q_bat), C_rep=float(self.config.C_rep),
            E_life=float(self.config.E_life), soc_min=float(self.config.soc_min), soc_max=float(self.config.soc_max),
            pb_vals=self.config.pb_vals, n_vals=self.config.n_vals, soc_vals=self.config.soc_vals,
            transition_row=transition_row, V_next=V_next, use_exact=use_exact, dSoC=dSoC, is_macro=self.is_macro
        )

        p_fc_locked = P_d_eval - best_pbatt if self.is_macro else state.P_d - best_pbatt
        
        self.current_step += 1
        return Action(n_modules=int(best_n), p_batt=best_pbatt, p_fc=p_fc_locked)