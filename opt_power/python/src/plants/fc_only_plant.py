# src/plants/fc_only_plant.py
class FuelCellOnlyPlant:
    """
    Represents the physical baseline vessel power plant configuration.
    Contains the core algebraic degradation and efficiency equations.
    """
    def __init__(self, config):
        self.config = config
        self.p_star = config.p_star
        self.k_s = config.k_s

def calculate_step_costs(self, P_d_step: float, n_step: int, n_prev: int) -> tuple[float, float]:
    if n_step <= 0:
        raise ValueError("Module allocation n must be strictly positive.")
        
    # Physical Constants (Table 1)
    k_H2 = 4.0          
    k_FC = 75000.0      
    tau_FC = 50000.0    
    a0, a1, a2 = 55.8460e-3, 10.0800e-3, 0.0556e-3 
    alpha_deg = 1.0

    # Power per active module
    p_module = P_d_step / n_step
    
    # 1. Operational Cost (H2 Consumption + FC Degradation)
    # H2 mass flow rate (g/s)
    m_H2 = a0 + a1 * p_module + a2 * (p_module ** 2)
    cost_h2_sec = k_H2 * (m_H2 / 1000.0)
    
    # Degradation rate (fraction of life lost per second)
    d_FC = (1.0 / (3600.0 * tau_FC)) * (1.0 + alpha_deg * ((p_module - self.p_star)**2) / (self.p_star**2))
    cost_deg_sec = k_FC * d_FC
    
    # Total operational cost over the macro-step (dt * lambda_scale)
    # Assuming Ts is your macro step size in seconds
    macro_step_sec = self.config.Ts 
    c_o = n_step * (cost_h2_sec + cost_deg_sec) * macro_step_sec
    
    # 2. Thermal/Mechanical Switching Stress Cost
    c_s = self.k_s * abs(n_step - n_prev)
    
    return c_o, c_s