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
    # S_max: float = 4000.0      # Maximum start/stop cycles before failure
    c_fc: float = 960          # FC module replacement cost [$/kW]
    k_h2: float = 4.0          # Hydrogen fuel cost [$/kg]
    alpha_fc: float = 1.0      # Degradation penalty factor for off-nominal loads
    delta_vswitch: float = 0.98e-6 # Switching voltage drop [V]
    delta_vlc: float = 1.79e-6 # Load-change voltage drop rate [V/kW]
    v_drop_max: float = 0.07   # Max permitted voltage drop (10% of nominal voltage)
    
    # Hydrogen Consumption Curve Coefficients (m_dot_H2 = a0 + a1*p + a2*p^2)
    # a0: float = 55.8460e-3     # [g/s]
    # a1: float = 10.0800e-3     # [g/(s*kW)]
    # a2: float = 0.0556e-3      # [g/(s*kW^2)]

    a0: float = 8.68055556e-02    # [g/s]
    a1: float = 9.54861111e-03     # [g/(s*kW)]
    a2: float = 5.42534722e-05    # [g/(s*kW^2)]
    
    # =========================================================================
    # 3. BATTERY PHYSICAL PARAMETERS (Hybrid Additions)
    # =========================================================================
    Q_bat: float = 500.0        # Nominal capacity of the battery pack [kWh]
    c_bat_kwh: float = 178.41   # Replacement cost per kWh [$/kWh]
    soc_min: float = 0.2       # Minimum safe State of Charge (20%)
    soc_max: float = 0.8       # Maximum safe State of Charge (80%)
    soc_initial: float = 0.5   # Starting State of Charge (50%)
    soc_target: float = 0.5    # Target terminal State of Charge
    
    pb_max: float = 1200.0   # Maximum discharge limit [kW] (Assumed 2C rate)
    pb_min: float = -1200.0  # Maximum charge limit [kW]
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
    N_n: int = 15          # Number of fuel cell modules on board
    N_soc: int = 61       # Grid resolution for SoC dimension (SoC Grid)
    N_pb: int = 61        # Grid resolution for P_batt dimension (P_batt grid)
    N_pfc: int = 21                    # Grid resolution for previous P_fc dimension

    use_smart_grid: bool = True
    dP: float = 200.0


    # =========================================================================
    # 6. DIRECTORY & PATH MANAGEMENT
    # =========================================================================
    data_dir: str = "../data/raw/"
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

        # --- DERIVED PHYSICAL CONSTANTS ---
        # Total fuel cell stacks replacement cost [$]
        object.__setattr__(self, 'k_fc', self.c_fc * self.p_max)

        # Total battery replacement cost [$]
        object.__setattr__(self, 'C_rep', self.Q_bat * self.c_bat_kwh)
        
        # Total lifetime energy throughput (Doubled for bidirectional wear calculation) [kWh]
        object.__setattr__(self, 'E_life', 2.0 * self.n_cycles_rated * self.dod_rated * self.Q_bat)

        # Financial penalty per kW of fuel cell power variation
        object.__setattr__(self, 'lambda_trans', self.delta_vlc * self.c_fc / self.v_drop_max)
        
        # # Maximum start/stop cycles before failure
        object.__setattr__(self, 'S_max', self.v_drop_max / self.delta_vswitch)

        # --- GRID GENERATION ---
        # 1D discrete array for the module count dimension
        object.__setattr__(self, 'n_vals', np.arange(0, self.N_n+1, dtype=np.int32))

        # Absolute maximum power the system expects to handle
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
            
            # 3. Congruent SoC Grid rigidly derived from dP and Ts
            dSoC = (self.dP * self.Ts) / (self.Q_bat * 3600.0)
            soc_grid = np.arange(self.soc_min, self.soc_max + (dSoC / 2.0), dSoC) # dSoC/2 prevents float cutoff
            object.__setattr__(self, 'soc_vals', soc_grid)
            object.__setattr__(self, 'N_soc', len(soc_grid))

            # 4. Snap Initial & Target SOC to the lattice
            idx_init = int(np.round((self.soc_initial - self.soc_min) / dSoC))
            object.__setattr__(self, 'soc_initial', float(soc_grid[idx_init]))
            
            if hasattr(self, 'soc_target'):
                idx_tgt = int(np.round((self.soc_target - self.soc_min) / dSoC))
                object.__setattr__(self, 'soc_target', float(soc_grid[idx_tgt]))
            
            object.__setattr__(self, 'N_Pd', int(np.ceil(max_fc_power / self.dP)))
            
        else:
            # Legacy Manual Grids
            object.__setattr__(self, 'soc_vals', np.linspace(self.soc_min, self.soc_max, self.N_soc))
            object.__setattr__(self, 'pb_vals', np.linspace(self.pb_min, self.pb_max, self.N_pb))
            object.__setattr__(self, 'pfc_vals', np.linspace(0.0, max_fc_power, self.N_pfc))
            
            
            
        # Fire the diagnostic tracker
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
            
        if self.Ts % self.dt_sim != 0:
            raise ValueError("Macro time step Ts must be a perfect multiple of dt_sim.")
            
        if not (0.0 <= self.soc_min < self.soc_max <= 1.0):
            raise ValueError("Invalid battery SoC boundary definitions.")
              
        if not (self.soc_min <= self.soc_target <= self.soc_max):
            raise ValueError("Terminal soc_target must remain within the physical soc_min and soc_max boundaries.")
            
        if self.pb_max <= 0 or self.pb_min >= 0:
            raise ValueError("Battery discharge limit must be > 0, and charge limit must be < 0.")
        

    def _print_complexity_diagnostics(self, N_Pd_est):
        """Prints the Big-O complexity for the Augmented Hybrid solver upon initialization."""
        N_n_len = len(self.n_vals)
        
        # O(S) = N_d * N_n * N_soc * N_pfc
        S_nodes = N_Pd_est * N_n_len * self.N_soc * self.N_pfc
        
        # O(A) = N_n * N_bat
        A_nodes = N_n_len * self.N_pb
        
        # O(Comp) = T * |S| * (|N_d| + |A|)
        transitions = self.Ts * S_nodes * (N_Pd_est + A_nodes)
        
        # Est memory: 4D Value (float64=8), Policy N (int32=4), Policy Pbatt (float64=8) per time step
        bytes_per_state = 20
        memory_mb = ((self.Ts + 1) * S_nodes * bytes_per_state) / (1024 * 1024)
        
        print("\n" + "="*55)
        print(f"⚙️  EMS Configuration Loaded | Smart Grid: {'ON' if self.use_smart_grid else 'OFF'}")
        print("="*55)
        if self.use_smart_grid:
            print(f" -> dP Step Size       : {self.dP} kW")
        print(f" -> Grid Dimensions    : N_d≈{N_Pd_est}, N_n={N_n_len}, N_soc={self.N_soc}, N_fc={self.N_pfc}, N_bat={self.N_pb}")
        print(f" -> Augmented Space |S|: {S_nodes:,} states")
        print(f" -> Action Space |A|   : {A_nodes:,} actions")
        print(f" -> Est. Big-O Comput. : O({transitions:,}) operations")
        print(f" -> Est. RAM Footprint : ~{memory_mb:.2f} MB")
        print("="*55 + "\n")
        
        

        