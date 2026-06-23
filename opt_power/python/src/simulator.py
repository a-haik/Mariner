# src/simulator.py
import numpy as np
from config import SimConfig

class Simulator:
    """
    Unified execution engine that coordinates the interaction between
    an online Controller strategy and a physical Plant model.
    """
    def __init__(self, config: SimConfig, P_d: np.ndarray, plant):
        self.config = config
        self.P_d = P_d.flatten()
        self.T = len(self.P_d)
        self.plant = plant  # Injected plant hardware abstraction layer
        
        # Trajectory historical monitoring caches
        self.n = None
        self.C_o = None
        self.C_s = None
        self.C = None

    def run(self, controller) -> float:
        """
        Drives the sequential execution loop step-by-step.
        """
        # 1. Obtain the full sequence of module decisions from the controller
        n_decisions = controller.compute(self.P_d, self.config.n0)
        
        # 2. Pre-allocate tracking vectors
        C_o_vec = np.zeros(self.T)
        C_s_vec = np.zeros(self.T)
        
        # 3. Step through time tracking system interactions
        n_prev = self.config.n0
        for t in range(self.T):
            n_curr = n_decisions[t]
            
            # Request physical consequences from our hardware plant wrapper
            c_o, c_s = self.plant.calculate_step_costs(self.P_d[t], n_curr, n_prev)
            
            # In the baseline MATLAB code, the initial cycle cost at t=0 is forced to 0
            if t == 0:
                c_s = 0.0
                
            C_o_vec[t] = c_o
            C_s_vec[t] = c_s
            n_prev = n_curr
            
        # Save tracking data arrays for plotting utilities
        self.n = n_decisions
        self.C_o = C_o_vec
        self.C_s = C_s_vec
        self.C = C_o_vec + C_s_vec
        
        return float(np.sum(self.C))