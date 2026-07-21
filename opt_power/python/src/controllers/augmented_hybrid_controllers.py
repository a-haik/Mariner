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
    calc_cost_transient,
    get_exact_index_1d,
    calculate_continuous_bounds,
    gss_augmented
)

class AugmentedFCLockedControl(ControlLaw):
    """
    Simulates real-world hardware limits (PLCs). Snaps continuous reality to 
    the 4D discrete SDP grid, extracting a rigid fuel cell setpoint.
    """
    def __init__(self, config: SimConfig, p_grid: np.ndarray, n_vals: np.ndarray, soc_vals: np.ndarray, 
                 pfc_vals: np.ndarray, policy_n: np.ndarray, policy_pbatt: np.ndarray):
        self.config = config
        self.p_grid = p_grid
        self.n_vals = n_vals
        self.soc_vals = soc_vals
        self.pfc_vals = pfc_vals
        self.policy_n = policy_n
        self.policy_pbatt = policy_pbatt
        
        # --- ZOH CACHE VARIABLES ---
        self.last_macro_idx = -1
        self.locked_action = None

    def get_action(self, state: State, time_sec: float) -> Action:
        
        macro_idx = int(time_sec // self.config.Dt)
        
        # ZERO-ORDER HOLD: Only evaluate at the start of a new 300s macro-step!
        if macro_idx > self.last_macro_idx:
            self.last_macro_idx = macro_idx
            t_idx = min(macro_idx, len(self.policy_n) - 1)
            
            # 1. Snap 4D continuous sensors to discrete grid
            idx_p = nearest_index_1d(self.p_grid, state.P_d)
            idx_n = nearest_index_1d(self.n_vals, float(state.n_prev))
            idx_soc = nearest_index_1d(self.soc_vals, state.soc)
            idx_pfc = nearest_index_1d(self.pfc_vals, state.p_fc_prev)
            
            # 2. Extract discrete intents
            best_n_idx = self.policy_n[t_idx, idx_p, idx_n, idx_soc, idx_pfc]
            n_opt = int(self.n_vals[best_n_idx])
            p_batt_disc = float(self.policy_pbatt[t_idx, idx_p, idx_n, idx_soc, idx_pfc])
            
            # 3. Calculate Intended Rigid FC Setpoint
            discrete_p_d = self.p_grid[idx_p]
            p_fc_locked = discrete_p_d - p_batt_disc
            
            if n_opt == 0:
                p_fc_locked = 0.0
            else:
                p_fc_locked = max(0.0, p_fc_locked)
            
            # Lock the intent!
            self.locked_action = Action(n_modules=n_opt, p_batt=p_batt_disc, p_fc=p_fc_locked)
        
        # For the next 299 seconds, the exact same locked P_fc is returned.
        # The plant will dynamically route the 1Hz P_d noise into the battery!
        return self.locked_action


class AugmentedPolicyControl(ControlLaw):
    """
    Smooths out the discrete policy matrices via 2D Bilinear Interpolation.
    """
    def __init__(self, config: SimConfig, p_grid: np.ndarray, n_vals: np.ndarray, soc_vals: np.ndarray, 
                 pfc_vals: np.ndarray, policy_n: np.ndarray, policy_pbatt: np.ndarray, is_macro: bool = False):
        self.config = config
        self.is_macro = is_macro
        self.p_grid = p_grid
        self.n_vals = n_vals
        self.soc_vals = soc_vals
        self.pfc_vals = pfc_vals
        self.policy_n = policy_n
        self.policy_pbatt = policy_pbatt
        
        # --- ZOH CACHE VARIABLES ---
        self.last_macro_idx = -1
        self.locked_action = None

    def get_action(self, state: State, time_sec: float) -> Action:

        macro_idx = int(time_sec // self.config.Dt)
        
        # ZERO-ORDER HOLD
        if macro_idx > self.last_macro_idx:
            self.last_macro_idx = macro_idx
            t_idx = min(macro_idx, len(self.policy_n) - 1)

            idx_p = nearest_index_1d(self.p_grid, state.P_d)
            idx_n = nearest_index_1d(self.n_vals, float(state.n_prev))
            idx_soc = nearest_index_1d(self.soc_vals, state.soc)
            idx_pfc = nearest_index_1d(self.pfc_vals, state.p_fc_prev)
            
            best_n_idx = self.policy_n[t_idx, idx_p, idx_n, idx_soc, idx_pfc]
            n_opt = int(self.n_vals[best_n_idx])
            
            use_sg = getattr(self.config, 'use_smart_grid', False)
            if self.is_macro and use_sg:
                p_batt_opt = float(self.policy_pbatt[t_idx, idx_p, idx_n, idx_soc, idx_pfc])
            else:
                slice_2d = self.policy_pbatt[t_idx, idx_p, idx_n, :, :] 
                p_batt_opt = bilinear_interp_2d(self.soc_vals, self.pfc_vals, slice_2d, state.soc, state.p_fc_prev)
            
            p_fc_locked = state.P_d - p_batt_opt

            if n_opt == 0:
                p_fc_locked = 0.0
            else:
                p_fc_locked = max(0.0, p_fc_locked)
            
            # Lock the intent!
            self.locked_action = Action(n_modules=n_opt, p_batt=p_batt_opt, p_fc=p_fc_locked)
        
        return self.locked_action

@njit(cache=True)
def _run_augmented_1_step_lookahead(P_d_real: float, soc_real: float, n_prev: int, pfc_prev: float, Dt: float, 
                                    p_max: float, p_nom: float, k_fc: float, k_h2: float, S_max: float, 
                                    tau_fc: float, alpha_fc: float, a0: float, a1: float, a2: float,
                                    Q_bat: float, C_rep: float, E_life: float, lambda_trans: float, 
                                    soc_min: float, soc_max: float, pb_vals: np.ndarray, n_vals: np.ndarray, 
                                    soc_vals: np.ndarray, pfc_vals: np.ndarray,
                                    transition_row: np.ndarray, V_next: np.ndarray,
                                    use_exact: bool, dSoC: float, dP: float, is_macro: bool):
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

    pb_min_config = pb_vals[0]
    pb_max_config = pb_vals[-1]

    for a_idx in range(n_size):
        n_curr = n_vals[a_idx]
        
        if is_macro:
            for pbatt in pb_vals:

                epsilon = 1e-5

                # Kinematics
                soc_next = soc_real - (pbatt * (Dt / 3600.0)) / Q_bat
                if soc_next < soc_min - epsilon or soc_next > soc_max + epsilon:
                    continue

                p_fc_curr = P_d_real - pbatt
                
                # Constraints
                if p_fc_curr < -epsilon:
                    continue
                if n_curr > 0 and (p_fc_curr / n_curr) > p_max + epsilon:
                    continue
                if n_curr == 0 and p_fc_curr > epsilon:
                    continue

                # Centralized Cost Engine
                c_o = calc_cost_operational(n_curr, p_fc_curr, p_nom, tau_fc, alpha_fc, k_fc, k_h2, a0, a1, a2, Dt)
                c_s = calc_cost_switching(n_curr, n_prev, k_fc, S_max)
                c_bat = calc_cost_battery(pbatt, Dt, C_rep, E_life)
                c_trans = calc_cost_transient(n_curr, n_prev, p_fc_curr, pfc_prev, lambda_trans)

                # 2D Interpolation over expected future surface
                z_mat = exp_v[a_idx, :, :]

                if use_exact:
                    s_idx = get_exact_index_1d(soc_next, soc_min, dSoC, soc_size - 1)
                    p_idx = get_exact_index_1d(p_fc_curr, 0.0, dP, pfc_size - 1)
                    exp_future = z_mat[s_idx, p_idx]
                else:
                    exp_future = bilinear_interp_2d(soc_vals, pfc_vals, z_mat, soc_next, p_fc_curr)

                total_cost = c_o + c_s + c_bat + c_trans + exp_future
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_n = n_curr
                    best_pbatt = pbatt

        else:
            # CONTINUOUS OPTIMIZER LOGIC
            pb_min, pb_max = calculate_continuous_bounds(P_d_real, soc_real, n_curr, Dt, p_max, Q_bat, soc_min, soc_max, pb_min_config, pb_max_config)
            
            if pb_min > pb_max + 1e-5:
                continue
                
            opt_pb, total_cost = gss_augmented(pb_min, pb_max, 1.0, P_d_real, soc_real, n_curr, n_prev, pfc_prev, Dt, 
                                               p_max, p_nom, k_fc, k_h2, S_max, tau_fc, alpha_fc, a0, a1, a2, 
                                               Q_bat, C_rep, E_life, lambda_trans, soc_vals, pfc_vals, exp_v[a_idx, :, :])
            if total_cost < best_cost:
                best_cost = total_cost
                best_n = n_curr
                best_pbatt = opt_pb

    return best_n, best_pbatt

class AugmentedValueControl(ControlLaw):
    """
    Bypasses the Policy Matrix entirely. Solves the 4D Bellman equation online for the 
    exact continuous state, using the Value Matrix only for future expectations.
    """
    def __init__(self, config: SimConfig, p_grid: np.ndarray, transition_matrix: np.ndarray, 
                 V_matrix: np.ndarray, is_macro: bool = False):
        self.config = config
        self.p_grid = p_grid
        self.transition_matrix = transition_matrix
        self.V = V_matrix
        self.is_macro = is_macro
        
        # --- ZOH CACHE VARIABLES ---
        self.last_macro_idx = -1
        self.locked_n = 0
        self.locked_pfc = 0.0

    def get_action(self, state: State, time_sec: float) -> Action:

        macro_idx = int(time_sec // self.config.Dt)
        
        # 1. THE EMS (MACRO-STEP OPTIMIZATION)
        if macro_idx > self.last_macro_idx:
            self.last_macro_idx = macro_idx
            t_idx = min(macro_idx, len(self.V) - 1)
            
            if t_idx >= len(self.V) - 1:
                V_next = np.zeros((len(self.p_grid), len(self.config.n_vals), len(self.config.soc_vals), len(self.config.pfc_vals)))
            else:
                V_next = self.V[t_idx + 1]

            if self.is_macro:
                P_d_eval = self.p_grid[nearest_index_1d(self.p_grid, state.P_d)]
                n_eval = int(self.config.n_vals[nearest_index_1d(self.config.n_vals, float(state.n_prev))])
                soc_eval = self.config.soc_vals[nearest_index_1d(self.config.soc_vals, state.soc)]
                pfc_eval = self.config.pfc_vals[nearest_index_1d(self.config.pfc_vals, state.p_fc_prev)]
            else:
                P_d_eval = state.P_d
                n_eval = state.n_prev
                soc_eval = state.soc
                pfc_eval = state.p_fc_prev

            idx_p = nearest_index_1d(self.p_grid, P_d_eval)
            transition_row = self.transition_matrix[idx_p, :]

            use_sg = getattr(self.config, 'use_smart_grid', False)
            use_exact = use_sg and self.is_macro
            dP = getattr(self.config, 'dP', 140.0)
            dSoC = (dP * float(self.config.Dt)) / (float(self.config.Q_bat) * 3600.0) if use_sg else 0.0

            best_n, best_pbatt = _run_augmented_1_step_lookahead(
                P_d_real=P_d_eval, soc_real=soc_eval, n_prev=n_eval, pfc_prev=pfc_eval, Dt=float(self.config.Dt),
                p_max=float(self.config.p_max), p_nom=float(self.config.p_nom), k_fc=float(self.config.k_fc),
                k_h2=float(self.config.k_h2), S_max=float(self.config.S_max), tau_fc=float(self.config.tau_fc),
                alpha_fc=float(self.config.alpha_fc), a0=float(self.config.a0), a1=float(self.config.a1),
                a2=float(self.config.a2), Q_bat=float(self.config.Q_bat), C_rep=float(self.config.C_rep),
                E_life=float(self.config.E_life), lambda_trans=float(self.config.lambda_trans),
                soc_min=float(self.config.soc_min), soc_max=float(self.config.soc_max),
                pb_vals=self.config.pb_vals, n_vals=self.config.n_vals, soc_vals=self.config.soc_vals, 
                pfc_vals=self.config.pfc_vals, transition_row=transition_row, V_next=V_next,
                use_exact=use_exact, dSoC=dSoC, dP=dP, is_macro=self.is_macro
            )

            # Determine Strategic Intent 
            p_fc_intent = P_d_eval - best_pbatt if self.is_macro else state.P_d - best_pbatt
            
            # LOCK THE FUEL CELL INTENT
            self.locked_n = int(best_n)
            if self.locked_n == 0:
                self.locked_pfc = 0.0
            else:
                self.locked_pfc = max(0.0, p_fc_intent)

        # 2. THE PMS (MICRO-STEP TACTICAL EXECUTION)
        # The Battery constantly calculates the remainder to absorb 1Hz noise
        p_batt_dynamic = state.P_d - self.locked_pfc

        return Action(n_modules=self.locked_n, p_batt=p_batt_dynamic, p_fc=self.locked_pfc)