# python/src/plants/augmented_hybrid_plant.py
from typing import Tuple, Dict
from src.config import SimConfig
from src.core import State, Action
from src.plants.base import BasePlant
from src.utils.math_utils import (
    calc_cost_operational, 
    calc_cost_switching, 
    calc_cost_transient
)

class AugmentedFuelCellOnlyPlant(BasePlant):
    """
    Continuous-time physical simulation of a Fuel Cell ONLY system.
    Upgraded to track the 4D state transitions and penalize the massive 
    high-frequency transient load variations that occur without a battery.
    """
    def __init__(self, config: SimConfig):
        self.config = config

    def step(self, state: State, action: Action, dt: float) -> Tuple[State, Dict[str, float]]:
        n_active = action.n_modules
        
        # In a purely FC system, the fuel cells MUST absorb the raw, exact continuous demand
        p_fc_actual = state.P_d

        # Centralized Cost Engine Calculations (No battery cost)
        c_o_fc = calc_cost_operational(
            n_active=n_active, p_fc=p_fc_actual, p_nom=self.config.p_nom, 
            tau_fc=self.config.tau_fc, alpha_fc=self.config.alpha_fc, 
            k_fc=self.config.k_fc, k_h2=self.config.k_h2, 
            a0=self.config.a0, a1=self.config.a1, a2=self.config.a2, dt=dt
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

        # Construct Next State (SoC remains entirely static)
        next_state = State(
            P_d=0.0, 
            n_prev=n_active, 
            soc=state.soc, 
            p_fc_prev=p_fc_actual
        )
        
        # Telemetry matches the dictionary shape of the Hybrid plant for the Plotter
        telemetry = {
            'p_fc_actual': p_fc_actual,
            'p_batt_actual': 0.0,
            'soc': state.soc,
            'n_active': n_active,
            'cost_o': c_o_fc,
            'cost_bat': 0.0,
            'cost_s': c_s,
            'cost_tr': c_trans,
            'cost_total': c_o_fc + c_s + c_trans
        }
        
        return next_state, telemetry