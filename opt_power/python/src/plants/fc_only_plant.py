# python/src/plants/fc_only_plant.py
from typing import Tuple, Dict
from config import SimConfig
from src.core import State, Action
from src.plants.base import BasePlant

class FuelCellOnlyPlant(BasePlant):
    """
    Physical baseline vessel power plant configuration (No Battery).
    Evaluates costs in continuous time based on actual electrochemical equations.
    """
    def __init__(self, config: SimConfig):
        self.config = config

    def step(self, state: State, action: Action, dt: float) -> Tuple[State, Dict[str, float]]:
        n_active = action.n_modules
        p_fc_total = state.P_d
        
        # Prevent division by zero if controller turns everything off
        if n_active <= 0:
            c_o = float('inf')
            p_module = 0.0
        else:
            p_module = p_fc_total / n_active
            
            # 1. Hydrogen Consumption Rate [g/s]
            m_dot_h2 = (self.config.a0 + 
                        self.config.a1 * p_module + 
                        self.config.a2 * (p_module ** 2))
            
            # 2. Continuous Degradation Rate [1/s]
            d_fc = (1.0 / (3600.0 * self.config.tau_fc)) * (
                1.0 + self.config.alpha_fc * ((p_module - self.config.p_nom) ** 2) / (self.config.p_nom ** 2)
            )
            
            # 3. Total Operating Cost Rate [€/s] 
            # Note: k_h2 is in €/kg, so we divide m_dot_h2 by 1000
            c_o_rate = (self.config.k_h2 * m_dot_h2 / 1000.0) + (self.config.k_fc * d_fc)
            
            # MATH FIX: Integrate the continuous rate over the time step dt
            c_o = n_active * c_o_rate * dt

        # 4. Discrete Switching Cost [€]
        k_s = self.config.k_fc / self.config.S_max
        c_s = k_s * abs(n_active - state.n_prev)
        
        # Prepare the next state (SoC remains constant since there is no battery)
        next_state = State(P_d=0.0, n_prev=n_active, soc=state.soc)
        
        # Pack telemetry for the agnostic visualization script
        telemetry = {
            'p_fc_total': p_fc_total,
            'n_active': n_active,
            'cost_o': c_o,
            'cost_s': c_s,
            'cost_total': c_o + c_s
        }
        
        return next_state, telemetry