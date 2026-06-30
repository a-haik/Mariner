# python/src/plants/fc_only_plant.py
from typing import Tuple, Dict
from src.config import SimConfig
from src.core import State, Action
from src.plants.base import BasePlant

class FuelCellOnlyPlant(BasePlant):
    """
    Physical baseline (No Battery).
    Cleaned of continuous-time blackout penalties.
    """
    def __init__(self, config: SimConfig):
        self.config = config

    def step(self, state: State, action: Action, dt: float) -> Tuple[State, Dict[str, float]]:
        n_active = action.n_modules
        p_fc_actual = state.P_d  # <-- UNIFIED NOMENCLATURE
        
        if n_active <= 0:
            c_o = float('inf') if p_fc_actual > 0 else 0.0
            p_module = 0.0
        else:
            p_module = p_fc_actual / n_active
            
            # Exact Continuous Electrochemical Costs
            m_dot_h2 = (self.config.a0 + self.config.a1 * p_module + self.config.a2 * (p_module ** 2))
            d_fc = (1.0 / (3600.0 * self.config.tau_fc)) * (
                1.0 + self.config.alpha_fc * ((p_module - self.config.p_nom) ** 2) / (self.config.p_nom ** 2)
            )
            c_o_rate = (self.config.k_h2 * m_dot_h2 / 1000.0) + (self.config.k_fc * d_fc)
            c_o = n_active * c_o_rate * dt

        # Switching Cost
        k_s = self.config.k_fc / self.config.S_max
        c_s = k_s * abs(n_active - state.n_prev)
        
        next_state = State(P_d=0.0, n_prev=n_active, soc=state.soc)
        
        telemetry = {
            'p_fc_actual': p_fc_actual,
            'n_active': n_active,
            'cost_o': c_o,
            'cost_s': c_s,
            'cost_total': c_o + c_s
        }
        
        return next_state, telemetry