# python/src/plants/augmented_hybrid_plant.py
from typing import Tuple, Dict
from src.config import SimConfig
from src.core import State, Action
from src.plants.base import BasePlant
from src.utils.math_utils import (
    calc_cost_operational, 
    calc_cost_switching, 
    calc_cost_battery, 
    calc_cost_transient
)

class AugmentedHybridPlant(BasePlant):
    """
    Continuous-time physical simulation of the Hybrid FC/Battery system.
    Upgraded to track 4D state transitions (specifically previous FC power) 
    and penalize high-frequency transient load variations.
    """
    def __init__(self, config: SimConfig):
        self.config = config

    def step(self, state: State, action: Action, dt: float) -> Tuple[State, Dict[str, float]]:
        n_active = action.n_modules
        p_fc_target = action.p_fc

        # 1. Battery acts as the high-frequency buffer against the true continuous demand
        p_batt_real = state.P_d - p_fc_target

        # 2. Hardware Limits & Battery Fail-safes
        p_batt_real = max(self.config.pb_min, min(self.config.pb_max, p_batt_real))

        # SoC Limits (Prevent overcharge / overdischarge)
        if state.soc <= self.config.soc_min and p_batt_real > 0:
            p_batt_real = 0.0  # Force stop discharging
        elif state.soc >= self.config.soc_max and p_batt_real < 0:
            p_batt_real = 0.0  # Force stop charging

        # 3. Fail-safe Override: FC MUST absorb the remainder to prevent a blackout
        p_fc_actual = state.P_d - p_batt_real

        # 4. Centralized Cost Engine Calculations
        c_o_fc = calc_cost_operational(
            n_active=n_active, p_fc=p_fc_actual, p_nom=self.config.p_nom, 
            tau_fc=self.config.tau_fc, alpha_fc=self.config.alpha_fc, 
            k_fc=self.config.k_fc, k_h2=self.config.k_h2, 
            a0=self.config.a0, a1=self.config.a1, a2=self.config.a2, dt=dt
        )

        c_bat = calc_cost_battery(
            p_batt=p_batt_real, dt=dt, 
            C_rep=self.config.C_rep, E_life=self.config.E_life
        )

        c_s = calc_cost_switching(
            n_active=n_active, n_prev=state.n_prev, 
            k_fc=self.config.k_fc, S_max=self.config.S_max
        )

        c_trans = calc_cost_transient(
            n_curr=n_active, n_prev=state.n_prev, 
            p_fc_curr=p_fc_actual, p_fc_prev=state.p_fc_prev, 
            lambda_trans=self.config.lambda_trans
        )

        # 5. State Kinematics (Integrate State of Charge)
        soc_next = state.soc - (p_batt_real * (dt / 3600.0)) / self.config.Q_bat
        soc_next = max(0.0, min(1.0, soc_next)) # Absolute floating-point safety bound

        # 6. Construct Next State (Propagating the new p_fc_actual into p_fc_prev)
        next_state = State(
            P_d=0.0, 
            n_prev=n_active, 
            soc=soc_next, 
            p_fc_prev=p_fc_actual
        )
        
        # 7. Rich Telemetry for the Simulator
        telemetry = {
            'p_fc_actual': p_fc_actual,
            'p_batt_actual': p_batt_real,
            'soc': soc_next,
            'n_active': n_active,
            'cost_o': c_o_fc,
            'cost_bat': c_bat,
            'cost_s': c_s,
            'cost_tr': c_trans,
            'cost_total': c_o_fc + c_bat + c_s + c_trans
        }
        
        return next_state, telemetry