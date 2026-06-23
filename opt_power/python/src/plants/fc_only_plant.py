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
        """
        Calculates physical costs for an isolated discrete macro time interval.
        """
        if n_step <= 0:
            raise ValueError("Module allocation n must be strictly positive.")
            
        # 1. Operational Efficiency Deficit Cost
        c_o = ((P_d_step / self.p_star - n_step) ** 2) / n_step
        
        # 2. Thermal/Mechanical Switching Stress Cost
        c_s = self.k_s * abs(n_step - n_prev)
        
        return c_o, c_s