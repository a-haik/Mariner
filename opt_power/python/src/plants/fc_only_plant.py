# python/src/plants/fc_only_plant.py
from typing import Tuple, Dict
from src.config import SimConfig
from src.core import State, Action
from src.plants.base import BasePlant
from src.utils.math_utils import calc_cost_operational, calc_cost_switching

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
        
        # Exact Continuous Costs (Centralized Engine)
        c_o = calc_cost_operational(
            n_active=n_active, p_fc=p_fc_actual, p_nom=self.config.p_nom, 
            tau_fc=self.config.tau_fc, alpha_fc=self.config.alpha_fc, 
            k_fc=self.config.k_fc, k_h2=self.config.k_h2, 
            a0=self.config.a0, a1=self.config.a1, a2=self.config.a2, dt=dt
        )

        # Switching Cost
        c_s = calc_cost_switching(
            n_active=n_active, n_prev=state.n_prev, 
            k_fc=self.config.k_fc, S_max=self.config.S_max
        )
        
        next_state = State(P_d=0.0, n_prev=n_active, soc=state.soc)
        
        telemetry = {
            'p_fc_actual': p_fc_actual,
            'n_active': n_active,
            'cost_o': c_o,
            'cost_s': c_s,
            'cost_total': c_o + c_s
        }
        
        return next_state, telemetry