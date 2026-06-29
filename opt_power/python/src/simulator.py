# python/src/simulator.py
import numpy as np
from src.config import SimConfig
from src.core import State

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
        self.dt = dt_override if dt_override is not None else float(self.config.dt_sim)
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
            
            # PING THE CONTROLLER (Macro Time Step Only)
            if time_sec % self.config.Ts == 0:
                current_action = controller.get_action(current_state)
                
                if t == 0:
                    current_state.n_prev = current_action.n_modules

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
        return float(total_cost)