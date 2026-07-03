# python/src/plants/hybrid_plant.py
from typing import Tuple, Dict
from src.config import SimConfig
from src.core import State, Action
from src.plants.base import BasePlant
from src.utils.math_utils import calc_cost_operational, calc_cost_switching, calc_cost_battery

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
        p_batt_real = max(self.config.pb_min, min(self.config.pb_max, p_batt_real))

        # SoC Limits (Prevent overcharge / overdischarge)
        if state.soc <= self.config.soc_min and p_batt_real > 0:
            p_batt_real = 0.0  # Force stop discharging
        elif state.soc >= self.config.soc_max and p_batt_real < 0:
            p_batt_real = 0.0  # Force stop charging

        # 4. Fail-safe Override: FC MUST absorb the remainder to prevent a blackout
        p_fc_actual = state.P_d - p_batt_real

        # 5. Calculate Exact Continuous Costs
        
        # Fuel Cell Cost (H2 Consumption + Electrochemical Degradation)
        c_o_fc = calc_cost_operational(
            n_active=n_active, p_fc=p_fc_actual, p_nom=self.config.p_nom, 
            tau_fc=self.config.tau_fc, alpha_fc=self.config.alpha_fc, 
            k_fc=self.config.k_fc, k_h2=self.config.k_h2, 
            a0=self.config.a0, a1=self.config.a1, a2=self.config.a2, dt=dt
        )

        # Battery Wear Cost (Absolute Ah-throughput mapping)
        c_bat = calc_cost_battery(
            p_batt=p_batt_real, dt=dt, 
            C_rep=self.config.C_rep, E_life=self.config.E_life
        )

        # Switching Cost
        c_s = calc_cost_switching(
            n_active=n_active, n_prev=state.n_prev, 
            k_fc=self.config.k_fc, S_max=self.config.S_max
        )

        # 6. State Kinematics (Integrate State of Charge)
        soc_next = state.soc - (p_batt_real * (dt / 3600.0)) / self.config.Q_bat
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