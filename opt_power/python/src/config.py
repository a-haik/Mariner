# python/src/config.py
from dataclasses import dataclass, field
import numpy as np
import os

# =========================================================================
# THE PHYSICS CONFIG (Pure Math & Physics Only)
# =========================================================================
@dataclass(frozen=True)
class SimConfig:
    """
    Unified configuration parameters for the MARINER Optimal Power Distribution solver.
    Upgraded to support continuous-time physical simulation and Hybrid FC/Battery setups.
    """
    # =========================================================================
    # 1. TIME DOMAINS & DISCRETIZATION
    # =========================================================================
    dt: int = 1            # High-frequency physical simulation step [s]
    Dt: int = 300              # Macro-control decision interval / block-mean window [s]
    
    # =========================================================================
    # 2. FUEL CELL PHYSICAL PARAMETERS
    # =========================================================================
    p_max: float = 200.0       # Absolute ceiling power per PEMFC module [kW]
    p_nom: float = 80.0        # Nominal optimal load per PEMFC module [kW]

    n0: int = 0                # Initial number of active fuel cell modules
    nT: int = 0               # Target number of active modules at the end of the voyage
    
    # Degradation & Cost Coefficients
    tau_fc: float = 50000.0    # Expected service life at steady nominal operation [Hours]
    # S_max: float = 4000.0      # Maximum start/stop cycles before failure
    c_fc: float = 960          # FC module replacement cost [$/kW]
    k_h2: float = 4.0          # Hydrogen fuel cost [$/kg]
    alpha_fc: float = 1.0      # Degradation penalty factor for off-nominal loads
    delta_vswitch: float = 0.98e-6 # Switching voltage drop [V]
    delta_vlc: float = 1.79e-6 # Load-change voltage drop rate [V/kW]
    v_drop_max: float = 0.07   # Max permitted voltage drop (10% of nominal voltage)
    
    # Hydrogen Consumption Curve Coefficients (m_dot_H2 = a0 + a1*p + a2*p^2)

    # Values in draft
    # a0: float = 55.8460e-3     # [g/s]
    # a1: float = 10.0800e-3     # [g/(s*kW)]
    # a2: float = 5.56e-5      # [g/(s*kW^2)]

    # Values to reach 60% eff at 40kW and 55% eff at 80kW

    # Individual 200kW modules
    a0: float =1.01010101e-04    # [g/s]
    a1: float =8.83838384e-06    # [g/(s*kW)]
    a2: float =6.31313131e-08    # [g/(s*kW^2)]

    # =========================================================================
    # 3. BATTERY PHYSICAL PARAMETERS (Hybrid Additions)
    # =========================================================================
    Q_bat: float = 500.0        # Nominal capacity of the battery pack [kWh]
    c_bat_kwh: float = 178.41   # Replacement cost per kWh [$/kWh]
    soc_min: float = 0.2       # Minimum safe State of Charge (20%)
    soc_max: float = 0.8       # Maximum safe State of Charge (80%)
    soc_initial: float = 0.5   # Starting State of Charge (50%)
    soc_target: float = 0.5    # Target terminal State of Charge
    
    pb_max: float = 1000.0   # Maximum discharge limit [kW] (Assumed 2C rate)
    pb_min: float = -1000.0  # Maximum charge limit [kW]
    n_cycles_rated: float = 12000.0 # Manufacturer rated cycle life
    dod_rated: float = 0.8     # Depth of discharge for rated cycles

    # =========================================================================
    # 4. SIMULATION BOUNDARY CONDITIONS (Terminal Penalties)
    # =========================================================================
    apply_terminal_n_cost: bool = True  # Force FCs to shut down at time T (incurs final switching cost)
    apply_terminal_soc_cost: bool = True  # Apply symmetrical penalty/reward for final SoC deviation from soc_initial
    
    # =========================================================================
    # 5. MARKOV CHAIN & GRID CALIBRATION
    # =========================================================================

    alpha_mc: float = 0.5      # Dirichlet smoothing parameter for sparse transitions

    N_Pd: int = 6         # Number of discrete load levels (Power demand Grid)
    N_n: int = 16         # Number of fuel cell modules on board
    n_pack: int = 1       # Number of fuel cell modules in a pack

    N_soc: int = 25       # Grid resolution for SoC dimension (SoC Grid)
    N_pb: int = 15        # Grid resolution for P_batt dimension (P_batt grid)
    N_pfc: int = 20                    # Grid resolution for previous P_fc dimension

    use_smart_grid: bool = True
    dP: float = 200.0
    verbose: bool = True

    # Derived fields (populated automatically in __post_init__)
    C_rep: float = field(init=False)
    E_life: float = field(init=False)
    soc_vals: np.ndarray = field(init=False)

    def __post_init__(self):
        """Sanity check validations and derivation of mathematical constants."""
        # --- DERIVED PHYSICAL CONSTANTS ---
        object.__setattr__(self, 'k_fc', self.c_fc * self.p_max)
        object.__setattr__(self, 'C_rep', self.Q_bat * self.c_bat_kwh)
        object.__setattr__(self, 'E_life', 2.0 * self.n_cycles_rated * self.dod_rated * self.Q_bat)
        object.__setattr__(self, 'lambda_trans', self.delta_vlc * self.c_fc / self.v_drop_max)
        object.__setattr__(self, 'S_max', self.v_drop_max / self.delta_vswitch)

        # --- GRID GENERATION ---
        object.__setattr__(self, 'n_vals', np.arange(0, self.N_n+1, 2, dtype=np.int32))
        max_fc_power = self.N_n * self.p_max

        if self.use_smart_grid:
            # 1. Congruent Battery Power Grid
            pb_grid = np.arange(self.pb_min, self.pb_max + self.dP, self.dP)
            object.__setattr__(self, 'pb_vals', pb_grid)
            object.__setattr__(self, 'N_pb', len(pb_grid))
            
            # 2. Congruent FC Power Grid
            pfc_grid = np.arange(0.0, max_fc_power + self.dP, self.dP)
            object.__setattr__(self, 'pfc_vals', pfc_grid)
            object.__setattr__(self, 'N_pfc', len(pfc_grid))
            
            # 3. Congruent SoC Grid (Lattice Phasing: Anchored to soc_initial)
            dSoC = (self.dP * self.Dt) / (self.Q_bat * 3600.0)
            
            # Build bidirectional arrays outwards from the initial state
            upper_grid = np.arange(self.soc_initial, self.soc_max + 1e-6, dSoC)
            lower_grid = np.arange(self.soc_initial, self.soc_min - 1e-6, -dSoC)
            
            # Merge, drop the duplicated center node, and automatically sort
            soc_grid = np.unique(np.concatenate((lower_grid, upper_grid)))
            
            object.__setattr__(self, 'soc_vals', soc_grid)
            object.__setattr__(self, 'N_soc', len(soc_grid))

            # 4. Snap Target SOC to the newly phased lattice
            if hasattr(self, 'soc_target'):
                idx_tgt = (np.abs(soc_grid - self.soc_target)).argmin()
                object.__setattr__(self, 'soc_target', float(soc_grid[idx_tgt]))
            
        else:
            # Legacy Manual Grids
            object.__setattr__(self, 'soc_vals', np.linspace(self.soc_min, self.soc_max, self.N_soc))
            object.__setattr__(self, 'pb_vals', np.linspace(self.pb_min, self.pb_max, self.N_pb))
            object.__setattr__(self, 'pfc_vals', np.linspace(0.0, max_fc_power, self.N_pfc))
            
        # Fire the diagnostic tracker
        if self.verbose:
            self._print_complexity_diagnostics(self.N_Pd)

        # --- SANITY CHECKS ---
        if self.p_max <= 0 or self.p_nom <= 0:
            raise ValueError("Power limits p_max and p_nom must be strictly positive.")
            
        if self.n0 not in self.n_vals and self.n0 != 0:
            raise ValueError(f"Initial module state n0={self.n0} must fall within action space n_vals or be 0.")
        
        if self.nT not in self.n_vals and self.nT != 0:
            raise ValueError(f"Terminal module state nT={self.nT} must fall within action space n_vals or be 0.")
            
        if self.alpha_mc < 0:
            raise ValueError("Dirichlet smoothing coefficient alpha_mc cannot be negative.")
            
        if not (self.Dt / self.dt).is_integer():
            raise ValueError("Macro time step Dt must be a perfect multiple of dt.")
            
        if not (0.0 <= self.soc_min < self.soc_max <= 1.0):
            raise ValueError("Invalid battery SoC boundary definitions.")
              
        if not (self.soc_min <= self.soc_target <= self.soc_max):
            raise ValueError("Terminal soc_target must remain within the physical soc_min and soc_max boundaries.")
            
        if self.pb_max <= 0 or self.pb_min >= 0:
            raise ValueError("Battery discharge limit must be > 0, and charge limit must be < 0.")
        

    def _print_complexity_diagnostics(self, N_Pd):
        """Prints the Big-O complexity for the Augmented Hybrid solver upon initialization."""
        N_n_len = len(self.n_vals)
        
        # O(S) = N_d * N_n * N_soc * N_pfc
        S_nodes = N_Pd * N_n_len * self.N_soc * self.N_pfc
        
        # O(A) = N_n * N_bat
        A_nodes = N_n_len * self.N_pb
        
        # O(Comp) = T * |S| * (|N_d| + |A|)
        transitions = self.Dt * S_nodes * (N_Pd + A_nodes)
        
        # Est memory: 4D Value (float64=8), Policy N (int32=4), Policy Pbatt (float64=8) per time step
        bytes_per_state = 20
        memory_mb = ((self.Dt + 1) * S_nodes * bytes_per_state) / (1024 * 1024)
        
        print("\n" + "="*55)
        print(f"⚙️  EMS Configuration Loaded | Smart Grid: {'ON' if self.use_smart_grid else 'OFF'}")
        print("="*55)
        if self.use_smart_grid:
            print(f" -> dP Step Size       : {self.dP} kW")
        print(f" -> Grid Dimensions    : N_d={N_Pd}, N_n={N_n_len}, N_soc={self.N_soc}, N_fc={self.N_pfc}, N_bat={self.N_pb}")
        print(f" -> Augmented Space |S|: {S_nodes:,} states")
        print(f" -> Action Space |A|   : {A_nodes:,} actions")
        print(f" -> Est. Big-O Comput. : O({transitions:,}) operations")
        print(f" -> Est. RAM Footprint : ~{memory_mb:.2f} MB")
        print("="*55 + "\n")

# =========================================================================
# THE NEW ENVIRONMENT CONFIG (Paths & Directories Only)
# =========================================================================
@dataclass(frozen=True)
class EnvConfig:
    """
    Manages absolute paths and directory creation for the local machine.
    Kept strictly separate from physics parameters to ensure Vault hashing consistency.
    """
    data_dir_rel: str = "../data/raw/"
    vault_dir_rel: str = "../data/vault/"
    cache_dir_rel: str = "../data/cache/"

    # Derived absolute paths
    data_dir: str = field(init=False)
    vault_dir: str = field(init=False)
    cache_dir: str = field(init=False)

    def __post_init__(self):
        # Anchor the base directory to the 'python/' folder (one level up from src/)
        python_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        object.__setattr__(self, 'data_dir', os.path.abspath(os.path.join(python_dir, self.data_dir_rel)))
        object.__setattr__(self, 'vault_dir', os.path.abspath(os.path.join(python_dir, self.vault_dir_rel)))
        object.__setattr__(self, 'cache_dir', os.path.abspath(os.path.join(python_dir, self.cache_dir_rel)))
        
        # Ensure the directories actually exist before anything tries to write to them
        os.makedirs(self.vault_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)