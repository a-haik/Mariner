# python/src/config.py
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
    p_nom: float = 200.0       # Nominal optimal load per PEMFC module [kW]
    n0: int = 0                # Initial number of active fuel cell modules
    
    # Degradation & Cost Coefficients
    tau_fc: float = 50000.0    # Expected service life at steady nominal operation [Hours]
    S_max: float = 4000.0      # Maximum start/stop cycles before failure
    k_fc: float = 75000.0      # FC module replacement cost [€]
    k_h2: float = 4.0          # Hydrogen fuel cost [€/kg]
    alpha_fc: float = 1.0      # Degradation penalty factor for off-nominal loads
    
    # Hydrogen Consumption Curve Coefficients (m_dot_H2 = a0 + a1*p + a2*p^2)
    a0: float = 55.8460e-3     # [g/s]
    a1: float = 10.0800e-3     # [g/(s*kW)]
    a2: float = 0.0556e-3      # [g/(s*kW^2)]
    
    # =========================================================================
    # 3. BATTERY PHYSICAL PARAMETERS (Hybrid Additions)
    # =========================================================================
    C_bat: float = 2000.0        # Nominal capacity of the battery pack [kWh]
    c_bat_kwh: float = 250.0   # Replacement cost per kWh [€/kWh]
    soc_min: float = 0.2       # Minimum safe State of Charge (20%)
    soc_max: float = 0.8       # Maximum safe State of Charge (80%)
    soc_initial: float = 0.5   # Starting State of Charge (70%)
    
    pb_max: float = 6000.0   # Maximum discharge limit [kW] (Assumed 2C rate)
    pb_min: float = -6000.0  # Maximum charge limit [kW]
    n_cycles_rated: float = 12000.0 # Manufacturer rated cycle life
    dod_rated: float = 0.8     # Depth of discharge for rated cycles
    
    # =========================================================================
    # 4. MARKOV CHAIN & GRID CALIBRATION
    # =========================================================================

    alpha_mc: float = 0.5      # Dirichlet smoothing parameter for sparse transitions

    N_Pd: int = 12         # Number of discrete load levels (Power demand Grid)
    N_n: int = 4          # Number of fuel cell modules on board
    N_soc: int = 61       # Grid resolution for SoC dimension (SoC Grid)
    N_pb: int = 51        # Grid resolution for P_batt dimension (P_batt grid)

    # Derived fields (populated automatically in __post_init__)
    C_rep: float = field(init=False)
    E_life: float = field(init=False)
    soc_vals: np.ndarray = field(init=False)

    def __post_init__(self):
        """Sanity check validations and derivation of mathematical constants."""
        # --- DERIVED HYBRID CONSTANTS ---
        # 1D discrete array for the module count dimension
        object.__setattr__(self, 'n_vals', np.arange(1, self.N_n+1, dtype=np.int32))

        # 1D discrete array for the State of Charge dimension
        object.__setattr__(self, 'soc_vals', np.linspace(self.soc_min, self.soc_max, self.N_soc))

        # 1D discrete array for the State of Charge dimension
        object.__setattr__(self, 'pb_vals', np.linspace(self.pb_min, self.pb_max, self.N_pb))

        # Total battery replacement cost [€]
        object.__setattr__(self, 'C_rep', self.C_bat * self.c_bat_kwh)
        
        # Total lifetime energy throughput (Doubled for bidirectional wear calculation) [kWh]
        object.__setattr__(self, 'E_life', 2.0 * self.n_cycles_rated * self.dod_rated * self.C_bat)

        if self.p_max <= 0 or self.p_nom <= 0:
            raise ValueError("Power limits p_max and p_nom must be strictly positive.")
            
        if self.n0 not in self.n_vals and self.n0 != 0:
            raise ValueError(f"Initial module state n0={self.n0} must fall within action space n_vals or be 0.")
            
        if self.alpha_mc < 0:
            raise ValueError("Dirichlet smoothing coefficient alpha_mc cannot be negative.")
            
        if self.Ts % self.dt_sim != 0:
            raise ValueError("Macro time step Ts must be a perfect multiple of dt_sim.")
            
        if not (0.0 <= self.soc_min < self.soc_max <= 1.0):
            raise ValueError("Invalid battery SoC boundary definitions.")
            
        if self.pb_max <= 0 or self.pb_min >= 0:
            raise ValueError("Battery discharge limit must be > 0, and charge limit must be < 0.")
        
        

        