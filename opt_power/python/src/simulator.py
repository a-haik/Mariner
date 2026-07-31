# python/src/simulator.py
import numpy as np
from src.config import SimConfig
from src.core import State
from src.utils.math_utils import get_c_min_kwh

class Simulator:
    """
    Time-decoupled execution engine for the MARINER Optimal Power Distribution solver.
    """
    def __init__(self, config: SimConfig, P_d: np.ndarray, plant, dt_override: float = None):
        self.config = config
        self.P_d = P_d.flatten()
        self.T_sim = len(self.P_d)
        self.plant = plant
        
        # If dt_override is provided, use it; otherwise default to the 1Hz physics step
        self.dt = dt_override if dt_override is not None else float(self.config.dt)
        self.history = {}

    def run(self, controller) -> float:
        self.history = {'time': [], 'P_d': []}
        
        current_state = State(
            P_d=self.P_d[0], 
            n_prev=self.config.n0, 
            soc=self.config.soc_initial
        )
        current_action = None
        
        for t in range(self.T_sim):
            # Scale time based on the active dt
            time_sec = t * self.dt
            
            current_state.P_d = self.P_d[t]
            self.history['time'].append(time_sec)
            self.history['P_d'].append(self.P_d[t])
            
            # PING THE CONTROLLER
            current_action = controller.get_action(current_state, time_sec)

            # STEP THE PLANT
            current_state, telemetry = self.plant.step(
                state=current_state, 
                action=current_action, 
                dt=self.dt
            )
            
            for key, value in telemetry.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
                
        total_cost = sum(self.history.get('cost_total', [0.0]))
        terminal_n_cost = 0.0
        if self.config.apply_terminal_n_cost:
            k_s = self.config.k_fc / self.config.S_max
            # Penalize leaving modules on based on the target nT
            terminal_n_cost = k_s * abs(current_state.n_prev - self.config.nT)

        terminal_soc_cost = 0.0
        # Safely check if we are in the hybrid context
        if getattr(self.config, 'apply_terminal_soc_cost', False):
            # We need the minimum fuel cell generation cost to value the energy.
            # (Note: You may need to import _get_c_min_kwh or move it to a shared utils file)
            c_min_kwh = get_c_min_kwh(
                self.config.p_max, self.config.p_nom, self.config.tau_fc, 
                self.config.alpha_fc, self.config.k_fc, self.config.k_h2, 
                self.config.a0, self.config.a1, self.config.a2
            )
            
            # Positive cost for deficit, negative (reward) for surplus
            delta_e_kwh = (self.config.soc_target - current_state.soc) * self.config.Q_b
            terminal_soc_cost = delta_e_kwh * c_min_kwh

        # Add boundary conditions to the true operational cost
        total_cost += (terminal_n_cost + terminal_soc_cost)

        return {
            'total_cost': total_cost,
            'terminal_n_cost': terminal_n_cost,
            'terminal_soc_cost': terminal_soc_cost
        }