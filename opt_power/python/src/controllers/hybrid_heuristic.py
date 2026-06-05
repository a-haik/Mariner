# python/src/controllers/hybrid_heuristic.py
import numpy as np
from config import HybridSimConfig
from src.plants.hybrid_plant import _get_pfc_bounds

class HybridThresholdControl:
    """
    Closed-loop heuristic baseline for the Multi-Timescale system.
    Uses the statistical moving threshold for module switching, and a 
    charge-sustaining Proportional feedback loop for power splitting.
    """
    def __init__(self, config: HybridSimConfig, total_macro_steps: int, tau_relax: float = 5.0):
        self.config = config
        self.T_macro = total_macro_steps
        self.tau_relax = tau_relax
        
        # Calculate the dynamic Proportional Gain (K_p) [kW / %]
        # Represents power required to charge 1% SoC over a macro-step
        p_1_percent = (36.0 * self.config.e_bat) / (self.config.lambda_scale * self.config.dt)
        self.K_p = p_1_percent / self.tau_relax

    def get_action(self, macro_step_k: int, current_pd: float, 
                   n_prev: int, current_soc: float) -> tuple[int, float]:
        """Evaluates the 3-step heuristic logic for a single macro-step."""
        
        # --- STEP A: The Module Decision (n_k) ---
        # Identical to the baseline threshold statistics
        x = (current_pd / self.config.p_star) - n_prev
        t_rem = max(1.0, float(self.T_macro - macro_step_k - 1))
        x_thres = 2.0 * self.config.sigma * ((n_prev * self.config.k_s / t_rem) + 1.0)
        
        if x > x_thres:
            n_k = min(n_prev + 1, np.max(self.config.n_vals))
        elif x < -x_thres:
            n_k = max(n_prev - 1, np.min(self.config.n_vals))
        else:
            n_k = n_prev
            
        # --- STEP B: The Baseline Power Split & Feedback ---
        # Base power matches demand; feedback term pulls SoC back to target
        pfc_requested = current_pd + self.K_p * (self.config.soc_terminal_target - current_soc)
        
        # --- STEP C: The Charge-Sustaining Safety Net (Bounds Clipping) ---
        # 1. Evaluate physical battery limits over the next lambda window
        pfc_min, pfc_max = _get_pfc_bounds(current_soc, current_pd, self.config.e_bat, self.config.dt)
        
        # 2. Clip the requested power to strictly valid bounds
        best_pfc = np.clip(pfc_requested, pfc_min, pfc_max)
        
        # 3. Final sanity check: Fuel Cell cannot output negative power or exceed active module capacity
        best_pfc = np.clip(best_pfc, 0.0, n_k * self.config.p_star)
        
        return n_k, best_pfc