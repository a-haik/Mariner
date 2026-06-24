# python/config.py
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class SimConfig:
    """
    Unified configuration parameters for the MARINER Optimal Power Distribution solver.
    Upgraded to support continuous-time physical simulation and Hybrid FC/Battery setups.
    """
    # =========================================================================
    # 1. TIME DOMAINS & DISCRETIZATION
    # =========================================================================
    dt_sim: int = 1            # High-frequency physical simulation step [s]
    Ts: int = 300              # Macro-control decision interval / block-mean window [s]
    
    # =========================================================================
    # 2. FUEL CELL PHYSICAL PARAMETERS
    # =========================================================================
    p_max: float = 500.0       # Absolute ceiling power per PEMFC module [kW]
    p_nom: float = 200.0        # Nominal optimal load per PEMFC module [kW]
    n0: int = 2               # Initial number of active fuel cell modules
    
    # Degradation & Cost Coefficients (From Table 1)
    tau_fc: float = 50000.0    # Expected service life at steady nominal operation [Hours]
    S_max: float = 4000.0      # Maximum start/stop cycles before failure
    k_fc: float = 75000.0      # FC module replacement cost [€]
    k_h2: float = 4.0          # Hydrogen fuel cost [€/kg]
    alpha_fc: float = 1.0      # Degradation penalty factor for off-nominal loads
    
    # Hydrogen Consumption Curve Coefficients (ṁ_H2 = a0 + a1*p + a2*p^2)
    a0: float = 55.8460e-3     # [g/s]
    a1: float = 10.0800e-3     # [g/(s*kW)]
    a2: float = 0.0556e-3      # [g/(s*kW^2)]
    
    # =========================================================================
    # 3. BATTERY PHYSICAL PARAMETERS
    # =========================================================================
    C_bat: float = 25.0        # Nominal capacity of the lithium-ion battery pack [kWh]
    c_bat_kwh: float = 125.0   # Replacement cost per kWh [€/kWh]
    soc_min: float = 0.2       # Minimum safe State of Charge (20%)
    soc_max: float = 0.8       # Maximum safe State of Charge (80%)
    soc_initial: float = 0.7   # Starting State of Charge (70%)
    
    # =========================================================================
    # 4. MARKOV CHAIN (DTMC) CALIBRATION
    # =========================================================================
    n_states: int = 16          # Number of discrete load levels
    alpha_mc: float = 0.5      # Dirichlet smoothing parameter for sparse transitions
    
    # =========================================================================
    # 5. CONTROL ACTION SPACE
    # =========================================================================
    n_vals: np.ndarray = field(
        default_factory=lambda: np.arange(1, 5, dtype=np.int32)
    )

    def __post_init__(self):
        """Sanity check validations for physical constraints."""
        if self.p_max <= 0 or self.p_nom <= 0:
            raise ValueError("Power limits p_max and p_nom must be strictly positive.")
            
        if self.n0 not in self.n_vals:
            raise ValueError(f"Initial module state n0={self.n0} must fall within action space n_vals.")
            
        if self.alpha_mc < 0:
            raise ValueError("Dirichlet smoothing coefficient alpha_mc cannot be negative.")
            
        if self.Ts % self.dt_sim != 0:
            raise ValueError("Macro time step Ts must be a perfect multiple of dt_sim.")
            
        if not (0.0 <= self.soc_min < self.soc_max <= 1.0):
            raise ValueError("Invalid battery SoC boundary definitions.")