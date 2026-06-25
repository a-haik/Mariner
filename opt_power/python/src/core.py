# python/src/core.py
from dataclasses import dataclass

@dataclass
class State:
    """Represents the physical state of the system."""
    P_d: float          # Power demand [kW]
    n_prev: int         # Modules active in the previous time step
    soc: float          # Battery State of Charge [0.0 to 1.0]

@dataclass
class Action:
    """Represents the control decision requested by the EMS."""
    n_modules: int      # Discrete modules to turn on
    p_batt: float = 0.0 # Power requested from the battery [kW]