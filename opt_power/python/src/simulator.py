# python/src/simulator.py
import numpy as np
from config import SimConfig

class Simulator:
    """
    Evaluation plant framework that simulates power system execution, calculates
    operational/switching cost components, and tracks trajectories.
    Perfect replication of Simulator.m.
    """
    def __init__(self, config: SimConfig, P_d: np.ndarray):
        """
        Initializes the evaluation plant with design and demand configurations.
        
        Parameters:
            config: Unified system parameter dataclass instance.
            P_d: 1D array representing the continuous downsampled tracking power demand trajectory.
        """
        self.config = config
        self.P_d = P_d.flatten()
        self.T = len(self.P_d)      # Total number of macro tracking intervals
        
        # Historical metric tracking grids populated post-execution
        self.n = None               # Vector of allocated fuel cell modules
        self.C_o = None             # Time-series of operational costs incurred
        self.C_s = None             # Time-series of module switching penalties
        self.C = None               # Aggregated total cost profile trajectory

    def run(self, controller) -> float:
        """
        Executes the plant evaluation loop for a given controller structure.
        
        Parameters:
            controller: An instance of an abstract ControlLaw implementation.
            
        Returns:
            total_cost: Cumulative scalar cost evaluated over the entire simulation horizon.
        """
        # Execute the controller logic over the full timeline
        # Interface matches: controller.compute(P_d, n0)
        n_decision = controller.compute(self.P_d, self.config.n0)
        
        # Calculate resulting degradation and efficiency losses
        C_o, C_s, C, total_cost = self.calculate_cost(self.P_d, n_decision, self.config.k_s)
        
        # Retain trajectories internally for visualization export
        self.n = n_decision
        self.C_o = C_o
        self.C_s = C_s
        self.C = C
        
        return total_cost

    def calculate_cost(self, P_d: np.ndarray, n: np.ndarray, k_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Vectorized core cost equations matching the physical definitions of the system.
        
        Cost Formulation:
            C_o = ((P_d / p_star - n)^2) / n   (Operational & Fuel Consumption Loss)
            C_s = k_s * | diff(n) |             (Thermal / Voltage Switching Cycling Degradation)
        """
        # Element-wise division safety assertion: avoid division-by-zero if n contains zeros
        # Original logic assumes active modules n > 0 (n_vals starts at 1)
        if np.any(n <= 0):
            raise ValueError("Invalid control action encountered: module allocation n must be strictly positive.")
            
        # 1. Operational Cost tracking equation
        # Evaluates structural inefficiencies when modules deviate from their optimal efficiency point p_star
        C_o = ((P_d / self.config.p_star - n) ** 2) / n
        
        # 2. Switching Cost sequence alignment
        # In MATLAB, your supervisor used: n_diff = [0, diff(n)]
        # This explicitly anchors the initial cycle cost at t=0 to 0.
        n_diff = np.zeros_like(n, dtype=np.float64)
        n_diff[1:] = np.diff(n)
        C_s = k_s * np.abs(n_diff)
        
        # 3. Aggregate combined costs
        C = C_o + C_s
        total_cost = float(np.sum(C))
        
        return C_o, C_s, C, total_cost