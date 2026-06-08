# python/config.py
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class SimConfig:
    """
    Unified configuration parameters for the MARINER Optimal Power Distribution solver.
    Consolidates parameters across original exploratory scripts into a single source of truth.
    """
    # --- Plant & Cost Parameters ---
    k_s: float = 1.0           # Switching cost coefficient (degradation penalty)
    p_star: float = 200.0      # Reference power capacity per PEMFC module [kW]
    n0: int = 5                # Initial number of active fuel cell modules
    
    # --- Simulation Horizon Parameters ---
    Ts: int = 300              # Macro sample rate / block-mean aggregation window [s]
    num_runs: int = 1          # Number of sequential simulation runs
    enable_plotting: bool = True 
    
    # --- Markov Chain (DTMC) Calibration Parameters ---
    n_states: int = 8         # Number of discrete load levels 
    alpha: float = 0.5         # Dirichlet smoothing parameter for sparse transition count rows
    
    # --- Control Action Space ---
    n_vals: np.ndarray = field(
        default_factory=lambda: np.arange(1, 11, dtype=np.int32) 
    )
    
    # --- Synthetic Profile Parameters ---
    sigma: float = 0.5         # Standard deviation parameter for synthetic Gaussian random walks

    def __post_init__(self):
        """Sanity check validations for physical constraints."""
        if self.p_star <= 0:
            raise ValueError("Reference power p_star must be strictly positive.")
        if self.n0 not in self.n_vals:
            raise ValueError(f"Initial module state n0={self.n0} must fall within action space n_vals.")
        if self.alpha < 0:
            raise ValueError("Dirichlet smoothing coefficient alpha cannot be negative.")


@dataclass(frozen=True)
class HybridSimConfig(SimConfig):
    """
    Extended configuration for the Multi-Timescale Augmented SDP.
    Contains all tunable parameters for the continuous state and action spaces,
    as well as battery physics, guaranteeing zero hardcoded values in the solvers.
    """
    # --- Multi-Timescale Architecture ---
    dt: float = 5.0                # Micro-step resolution [s]
    lambda_scale: int = 60        # Macro-step multiplier (FC power locks for lambda * dt)
    mc_samples: int = 50          # Number of Monte Carlo paths for Variant C pre-computation

    # --- Discretization & Grid Resolutions (No Hardcoding) ---
    soc_step: float = 1.0          # Resolution of the SoC grid [%]
    p_fc_step: float = 100.0       # Resolution of the continuous P_fc grid [kW]
    
    # --- Battery Physics & Degradation ---
    e_bat: float = 10000.0          # Battery pack nominal energy capacity [kWh]
    c_bat_kwh: float = 178.41      # Cost of battery per kWh [$]
    n_eol_cycles: int = 3000       # Battery Cycle life to End-of-Life (EOL)
    
    # --- Boundary Conditions & Constraints ---
    soc_initial: float = 50.0      # Starting State of Charge [%]
    soc_terminal_target: float = 50.0 # Target minimum SoC at voyage completion [%]
    penalty_wall: float = 1e12     # Hard mathematical penalty for infeasible states / grid clipping

    def __post_init__(self):
        super().__post_init__()
        if self.lambda_scale <= 0:
            raise ValueError("lambda_scale must be a strictly positive integer.")