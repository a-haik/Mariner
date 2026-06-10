# python/config.py
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class SimConfig:
    """
    Unified configuration parameters for the MARINER Optimal Power Distribution solver.
    Consolidates macro-level constraints, legacy parameters, and micro-level battery physics
    into a single source of truth[cite: 223].
    """
    # --- Plant & Cost Parameters ---
    p_star: float = 200.0          # Reference power capacity per PEMFC module [kW] 
    n0: int = 5                    # Initial number of active fuel cell modules 
    s_max: int = 4000              # Max on/off switches expected in lifetime
    # s_max: int = 75000 
    k_h2: float = 4.0              # Cost of hydrogen fuel [€/kg] [cite: 225]
    k_fc: float = 75000.0          # Cost of FC stack replacement [€] [cite: 225]
    tau_fc: float = 50000.0        # Expected service life [Hours] [cite: 225]
    alpha_deg: float = 1.0         # Degradation acceleration factor [cite: 225]
    a0: float = 55.8460e-3         # H2 flow coefficient [g/s] [cite: 226]
    a1: float = 10.0800e-3         # H2 flow coefficient [g/(s kW)] [cite: 226]
    a2: float = 0.0556e-3          # H2 flow coefficient [g/(s kW^2)] [cite: 226]
    
    # k_s is marked as init=False since it is dynamically derived from Equation 23
    k_s: float = field(default=1.0, init=False) 
    
    # --- Legacy Simulation Parameters (Maintained for obsolete simulator.py) ---
    Ts: int = 300                  # Macro sample rate / block-mean aggregation window [s] [cite: 226]
    num_runs: int = 1              # Number of sequential simulation runs [cite: 227]
    enable_plotting: bool = True   # [cite: 227]
    
    # --- Markov Chain (DTMC) Calibration Parameters ---
    n_states: int = 8              # Number of discrete load levels [cite: 227]
    alpha: float = 0.5             # Dirichlet smoothing parameter for sparse transition count rows [cite: 228]
    
    # --- Control Action Space ---
    n_vals: np.ndarray = field(
        default_factory=lambda: np.arange(1, 11, dtype=np.int32) # [cite: 228]
    )
    sigma: float = 0.5             # Standard deviation parameter for synthetic Gaussian random walks [cite: 228]

    # --- Multi-Timescale Architecture (Hybrid) ---
    dt: float = 5.0                # Micro-step resolution [s] [cite: 229]
    lambda_scale: int = 60         # Macro-step multiplier (FC power locks for lambda * dt) [cite: 229]
    mc_samples: int = 50           # Number of Monte Carlo paths for Variant C pre-computation [cite: 229]

    # --- Discretization & Grid Resolutions (No Hardcoding) ---
    soc_step: float = 1.0          # Resolution of the SoC grid [%] [cite: 229]
    p_fc_step: float = 100.0       # Resolution of the continuous P_fc grid [kW] [cite: 230]
    
    # --- Battery Physics & Degradation ---
    e_bat: float = 10000.0         # Battery pack nominal energy capacity [kWh] [cite: 230]
    c_bat_kwh: float = 178.41      # Cost of battery per kWh [$] [cite: 230]
    n_eol_cycles: int = 3000       # Battery Cycle life to End-of-Life (EOL) [cite: 230]
    
    # --- Boundary Conditions & Constraints ---
    soc_initial: float = 50.0         # Starting State of Charge [%] [cite: 231]
    soc_terminal_target: float = 50.0 # Target minimum SoC at voyage completion [%] [cite: 231]
    penalty_wall: float = 1e12        # Hard mathematical penalty for infeasible states / grid clipping [cite: 231]

    def __post_init__(self):
        """Sanity check validations for physical constraints and dynamic coefficient math."""
        # EQUATION 23 IMPLEMENTATION: Compute the real-world switching penalty ($18.75 €/switch)
        # We must bypass the frozen dataclass write protection via object.__setattr__
        real_k_s = self.k_fc / self.s_max
        object.__setattr__(self, 'k_s', real_k_s)

        if self.p_star <= 0:
            raise ValueError("Reference power p_star must be strictly positive.") # [cite: 232]
        if self.n0 not in self.n_vals:
            raise ValueError(f"Initial module state n0={self.n0} must fall within action space n_vals.") # [cite: 232]
        if self.alpha < 0:
            raise ValueError("Dirichlet smoothing coefficient alpha cannot be negative.") # [cite: 232]
        if self.lambda_scale <= 0:
            raise ValueError("lambda_scale must be a strictly positive integer.") # [cite: 233]