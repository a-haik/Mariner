# python/src/plants/hybrid_plant.py
from typing import Tuple, Dict
from src.config import SimConfig
from src.core import State, Action
from src.plants.base import BasePlant

class HybridPlant(BasePlant):
    """
    Continuous-time physical simulation of the Hybrid Fuel Cell and Battery system.
    Implements high-frequency buffering, physical hardware limits, and exact degradation.
    """
    def __init__(self, config: SimConfig):
        self.config = config

    def step(self, state: State, action: Action, dt: float) -> Tuple[State, Dict[str, float]]:
        n_active = action.n_modules
        
        # 1. Read the rigid controller intent
        p_fc_target = action.p_fc

        # 2. Battery acts as the high-frequency buffer against the true continuous demand
        p_batt_real = state.P_d - p_fc_target

        # 3. Hardware Limits & Battery Fail-safes
        # Clamp to the physical C-rate boundaries
        p_batt_real = max(self.config.p_batt_min, min(self.config.p_batt_max, p_batt_real))

        # SoC Limits (Prevent overcharge / overdischarge)
        if state.soc <= self.config.soc_min and p_batt_real > 0:
            p_batt_real = 0.0  # Force stop discharging
        elif state.soc >= self.config.soc_max and p_batt_real < 0:
            p_batt_real = 0.0  # Force stop charging

        # 4. Fail-safe Override: FC MUST absorb the remainder to prevent a blackout
        p_fc_actual = state.P_d - p_batt_real

        # 5. Calculate Exact Continuous Costs
        # Fuel Cell Cost (H2 Consumption + Electrochemical Degradation)
        if n_active <= 0:
            c_o_fc = float('inf') if p_fc_actual > 0 else 0.0
            p_module = 0.0
        else:
            p_module = p_fc_actual / n_active
            
            m_dot_h2 = (self.config.a0 + self.config.a1 * p_module + self.config.a2 * (p_module ** 2))
            d_fc = (1.0 / (3600.0 * self.config.tau_fc)) * (
                1.0 + self.config.alpha_fc * ((p_module - self.config.p_nom) ** 2) / (self.config.p_nom ** 2)
            )
            c_o_rate = (self.config.k_h2 * m_dot_h2 / 1000.0) + (self.config.k_fc * d_fc)
            c_o_fc = n_active * c_o_rate * dt

        # Battery Wear Cost (Absolute Ah-throughput mapping)
        c_bat = self.config.C_rep * (abs(p_batt_real) * (dt / 3600.0)) / self.config.E_life

        # Switching Cost
        k_s = self.config.k_fc / self.config.S_max
        c_s = k_s * abs(n_active - state.n_prev)

        # 6. State Kinematics (Integrate State of Charge)
        soc_next = state.soc - (p_batt_real * (dt / 3600.0)) / self.config.C_bat
        soc_next = max(0.0, min(1.0, soc_next)) # Absolute floating-point safety bound

        next_state = State(P_d=0.0, n_prev=n_active, soc=soc_next)
        
        # 7. Rich Telemetry for the Simulator
        telemetry = {
            'p_fc_actual': p_fc_actual,
            'p_batt_actual': p_batt_real,
            'soc': soc_next,
            'n_active': n_active,
            'cost_o': c_o_fc,
            'cost_bat': c_bat,
            'cost_s': c_s,
            'cost_total': c_o_fc + c_bat + c_s
        }
        
        return next_state, telemetry