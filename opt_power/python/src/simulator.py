# python/src/simulator.py
import numpy as np
from config import SimConfig
from src.core import State, Action

class Simulator:
    """
    Time-decoupled execution engine for the MARINER Optimal Power Distribution solver.
    Implements a Zero-Order Hold (ZOH) loop to evaluate continuous high-frequency 
    physical consequences against macro-step (e.g., 300s) control decisions.
    """
    def __init__(self, config: SimConfig, P_d_continuous: np.ndarray, plant):
        self.config = config
        self.P_d = P_d_continuous.flatten()
        self.T_sim = len(self.P_d)
        self.plant = plant
        
        # Flexible telemetry dictionary to replace hardcoded arrays
        self.history = {}

    def run(self, controller) -> float:
        """
        Drives the sequential ZOH execution loop.
        """
        # Initialize tracking history
        self.history = {'time': [], 'P_d': []}
        
        # Initialize the physical state
        current_state = State(
            P_d=self.P_d[0], 
            n_prev=self.config.n0, 
            soc=self.config.soc_initial
        )
        
        # Placeholder for the ZOH control action
        current_action = None
        
        for t in range(self.T_sim):
            time_sec = t * self.config.dt_sim
            
            # 1. Update the state with the true high-frequency demand
            current_state.P_d = self.P_d[t]
            self.history['time'].append(time_sec)
            self.history['P_d'].append(self.P_d[t])
            
            # 2. PING THE CONTROLLER (Macro Time Step Only)
            if time_sec % self.config.Ts == 0:
                current_action = controller.get_action(current_state)
                
                # Bypass switching cost penalty for the initial startup at t=0
                if t == 0:
                    current_state.n_prev = current_action.n_modules

            # 3. STEP THE PLANT (High-Frequency Physical Simulation)
            current_state, telemetry = self.plant.step(
                state=current_state, 
                action=current_action, 
                dt=float(self.config.dt_sim)
            )
            
            # 4. LOG TELEMETRY (Dynamic mapping)
            for key, value in telemetry.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
                
        # The total simulation cost is the sum of all accumulated high-frequency costs
        total_cost = sum(self.history.get('cost_total', [0.0]))
        return float(total_cost)