# python/src/config.py
from dataclasses import dataclass, field
import numpy as np
import os

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
    p_max: float = 200.0       # Absolute ceiling power per PEMFC module [kW]
    p_nom: float = 80.0        # Nominal optimal load per PEMFC module [kW]
    n0: int = 0                # Initial number of active fuel cell modules
    nT: int = 0               # Target number of active modules at the end of the voyage
    
    # Degradation & Cost Coefficients
    tau_fc: float = 50000.0    # Expected service life at steady nominal operation [Hours]
    S_max: float = 4000.0      # Maximum start/stop cycles before failure
    c_fc: float = 750          # FC module replacement cost [€/kW]
    k_h2: float = 4.0          # Hydrogen fuel cost [€/kg]
    alpha_fc: float = 1.0      # Degradation penalty factor for off-nominal loads
    delta_vlc: float = 1.79e-6 # Voltage drop rate [V/kW]
    v_drop_max: float = 0.07   # Max permitted voltage drop (10% of nominal voltage)
    
    # Hydrogen Consumption Curve Coefficients (m_dot_H2 = a0 + a1*p + a2*p^2)
    a0: float = 55.8460e-3     # [g/s]
    a1: float = 10.0800e-3     # [g/(s*kW)]
    a2: float = 0.0556e-3      # [g/(s*kW^2)]
    
    # =========================================================================
    # 3. BATTERY PHYSICAL PARAMETERS (Hybrid Additions)
    # =========================================================================
    Q_bat: float = 2000.0        # Nominal capacity of the battery pack [kWh]
    c_bat_kwh: float = 250.0   # Replacement cost per kWh [€/kWh]
    soc_min: float = 0.2       # Minimum safe State of Charge (20%)
    soc_max: float = 0.8       # Maximum safe State of Charge (80%)
    soc_initial: float = 0.5   # Starting State of Charge (50%)
    soc_target: float = 0.5    # Target terminal State of Charge
    
    pb_max: float = 3000.0   # Maximum discharge limit [kW] (Assumed 2C rate)
    pb_min: float = -3000.0  # Maximum charge limit [kW]
    n_cycles_rated: float = 12000.0 # Manufacturer rated cycle life
    dod_rated: float = 0.8     # Depth of discharge for rated cycles

    # =========================================================================
    # 4. SIMULATION BOUNDARY CONDITIONS (Terminal Penalties)
    # =========================================================================
    apply_terminal_n_cost: bool = True   # Force FCs to shut down at time T (incurs final switching cost)
    apply_terminal_soc_cost: bool = True  # Apply symmetrical penalty/reward for final SoC deviation from soc_initial
    
    # =========================================================================
    # 5. MARKOV CHAIN & GRID CALIBRATION
    # =========================================================================

    alpha_mc: float = 0.5      # Dirichlet smoothing parameter for sparse transitions

    N_Pd: int = 8         # Number of discrete load levels (Power demand Grid)
    N_n: int = 16          # Number of fuel cell modules on board
    N_soc: int = 61       # Grid resolution for SoC dimension (SoC Grid)
    N_pb: int = 61        # Grid resolution for P_batt dimension (P_batt grid)
    N_pfc: int = 21                    # Grid resolution for previous P_fc dimension

    self.use_smart_grid = kwargs.get('use_smart_grid', False)

    # =========================================================================
    # 6. DIRECTORY & PATH MANAGEMENT
    # =========================================================================
    data_dir: str = "../data/"
    vault_dir: str = "../data/vault/"

    # Derived fields (populated automatically in __post_init__)
    C_rep: float = field(init=False)
    E_life: float = field(init=False)
    soc_vals: np.ndarray = field(init=False)

    def __post_init__(self):
        """Sanity check validations and derivation of mathematical constants."""
        # --- PATH RESOLUTION ---
        # Anchor the base directory to the 'python/' folder (one level up from src/)
        python_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        abs_data_dir = os.path.abspath(os.path.join(python_dir, self.data_dir))
        abs_vault_dir = os.path.abspath(os.path.join(python_dir, self.vault_dir))
        
        # Override the strings with absolute paths
        object.__setattr__(self, 'data_dir', abs_data_dir)
        object.__setattr__(self, 'vault_dir', abs_vault_dir)
        
        # Ensure the vault directory actually exists before anything tries to write to it
        os.makedirs(abs_vault_dir, exist_ok=True)

        # --- DERIVED HYBRID CONSTANTS ---
        # 1D discrete array for the module count dimension
        object.__setattr__(self, 'n_vals', np.arange(0, self.N_n+1, dtype=np.int32))

        # 1D discrete array for the State of Charge dimension
        object.__setattr__(self, 'soc_vals', np.linspace(self.soc_min, self.soc_max, self.N_soc))

        # 1D discrete array for the State of Charge dimension
        object.__setattr__(self, 'pb_vals', np.linspace(self.pb_min, self.pb_max, self.N_pb))

        # 1D discrete array for the previous Fuel Cell power dimension
        object.__setattr__(self, 'pfc_vals', np.linspace(0.0, self.N_n * self.p_max, self.N_pfc))

        # Total fuel cell stacks replacement cost [€]
        object.__setattr__(self, 'k_fc', self.c_fc * self.p_max)

        # Total battery replacement cost [€]
        object.__setattr__(self, 'C_rep', self.Q_bat * self.c_bat_kwh)
        
        # Total lifetime energy throughput (Doubled for bidirectional wear calculation) [kWh]
        object.__setattr__(self, 'E_life', 2.0 * self.n_cycles_rated * self.dod_rated * self.Q_bat)

        # Financial penalty per kW of fuel cell power variation
        object.__setattr__(self, 'lambda_trans', self.delta_vlc * self.c_fc / self.v_drop_max)

        if self.p_max <= 0 or self.p_nom <= 0:
            raise ValueError("Power limits p_max and p_nom must be strictly positive.")
            
        if self.n0 not in self.n_vals and self.n0 != 0:
            raise ValueError(f"Initial module state n0={self.n0} must fall within action space n_vals or be 0.")
        
        if self.nT not in self.n_vals and self.nT != 0:
            raise ValueError(f"Terminal module state nT={self.nT} must fall within action space n_vals or be 0.")

            
        if self.alpha_mc < 0:
            raise ValueError("Dirichlet smoothing coefficient alpha_mc cannot be negative.")
            
        if self.Ts % self.dt_sim != 0:
            raise ValueError("Macro time step Ts must be a perfect multiple of dt_sim.")
            
        if not (0.0 <= self.soc_min < self.soc_max <= 1.0):
            raise ValueError("Invalid battery SoC boundary definitions.")
              
        if not (self.soc_min <= self.soc_target <= self.soc_max):
            raise ValueError("Terminal soc_target must remain within the physical soc_min and soc_max boundaries.")
            
        if self.pb_max <= 0 or self.pb_min >= 0:
            raise ValueError("Battery discharge limit must be > 0, and charge limit must be < 0.")
        
        

        