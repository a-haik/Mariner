# python/src/simulator.py
import numpy as np
from config import HybridSimConfig
from src.plants.hybrid_plant import FuelCellBatteryPlant
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
    
class HybridSimulator:
    """
    Dual-timescale execution engine.
    Orchestrates the macro-level Controller and the micro-level Plant physics.
    """
    def __init__(self, config: HybridSimConfig, P_d_micro_profile: np.ndarray, plant: FuelCellBatteryPlant):
        self.config = config
        self.P_d = P_d_micro_profile.flatten()
        self.plant = plant
        
        # Timescale definitions
        self.T_micro = len(self.P_d)
        self.lambda_scale = self.config.lambda_scale
        self.T_macro = self.T_micro // self.lambda_scale
        
        # Continuous Trajectory Tracking (For Visualization)
        self.soc_history = np.zeros(self.T_micro + 1)
        self.soc_history[0] = self.config.soc_initial
        
        self.n_history = np.zeros(self.T_micro, dtype=np.int32)
        self.pfc_history = np.zeros(self.T_micro)
        self.pbat_history = np.zeros(self.T_micro)
        
        # Cost Tracking
        self.C_o_vec = np.zeros(self.T_macro)
        self.C_s_vec = np.zeros(self.T_macro)
        self.C_bat_vec = np.zeros(self.T_macro)

    def run(self, controller) -> float:
        """Executes the closed-loop simulation over the entire voyage."""
        n_prev = self.config.n0
        soc_curr = self.config.soc_initial
        
        total_voyage_cost = 0.0
        
        for k in range(self.T_macro):
            # 1. Sense: Read starting conditions for this macro-window
            micro_start_idx = k * self.lambda_scale
            micro_end_idx = micro_start_idx + self.lambda_scale
            
            # The demand the controller "sees" is the demand at the start of the window
            pd_curr = self.P_d[micro_start_idx]
            
            # 2. Decide: Query the policy tensors
            n_k, pfc_k = controller.get_action(k, pd_curr, n_prev, soc_curr)
            
            # 3. Act: Extract the true stochastic demand path and feed to physics plant
            p_d_micro_window = self.P_d[micro_start_idx:micro_end_idx]
            
            # Calculate physical degradation and final SoC over the lambda window
            macro_cost, soc_next = self.plant.calculate_macro_step(
                soc_curr, n_k, n_prev, pfc_k, p_d_micro_window
            )
            
            # Calculate individual cost components for plotting
            c_o = (((pfc_k / self.config.p_star) - n_k) ** 2) / n_k * self.lambda_scale
            c_s = self.config.k_s * abs(n_k - n_prev) if k > 0 else 0.0
            c_bat = macro_cost - c_o - c_s
            
            # 4. Record Trajectories
            self.C_o_vec[k] = c_o
            self.C_s_vec[k] = c_s
            self.C_bat_vec[k] = c_bat
            total_voyage_cost += macro_cost
            
            # Fill micro-step histories for the high-res visualization
            for t_offset in range(self.lambda_scale):
                t_global = micro_start_idx + t_offset
                p_bat_t = self.P_d[t_global] - pfc_k
                
                self.n_history[t_global] = n_k
                self.pfc_history[t_global] = pfc_k
                self.pbat_history[t_global] = p_bat_t
                
                # Re-integrate SoC strictly for plotting tracking
                delta_soc = - (p_bat_t * (self.config.dt / 3600.0) / self.config.e_bat) * 100.0
                self.soc_history[t_global + 1] = self.soc_history[t_global] + delta_soc

            # 5. Advance State
            n_prev = n_k
            soc_curr = soc_next
            
        return total_voyage_cost