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
    enable_plotting: bool = True #
    
    # --- Markov Chain (DTMC) Calibration Parameters ---
    n_states: int = 8         # Number of discrete load levels (M=16 in data execution block)
    alpha: float = 0.5         # Dirichlet smoothing parameter for sparse transition count rows
    
    # --- Control Action Space ---
    # Using field(default_factory=...) to generate mutable NumPy structures safely within a dataclass
    n_vals: np.ndarray = field(
        default_factory=lambda: np.arange(1, 11, dtype=np.int32) # Invariant action space: [1, 2, ..., 10]
    )
    
    # --- Synthetic Profile Parameters (Backward Compatibility) ---
    sigma: float = 0.5         # Standard deviation parameter for synthetic Gaussian random walks

    def __post_init__(self):
        """Sanity check validations for physical constraints."""
        if self.p_star <= 0:
            raise ValueError("Reference power p_star must be strictly positive.")
        if self.n0 not in self.n_vals:
            raise ValueError(f"Initial module state n0={self.n0} must fall within action space n_vals.")
        if self.alpha < 0:
            raise ValueError("Dirichlet smoothing coefficient alpha cannot be negative.")