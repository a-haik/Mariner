# python/src/core.py
from dataclasses import dataclass

@dataclass
class State:
    """
    Represents the physical state of the system at any given moment.
    """
    P_d: float          # Current power demand [kW]
    n_prev: int         # Number of active modules in the previous time step
    soc: float          # Battery State of Charge (SoC) [0.0 to 1.0]

@dataclass
class Action:
    """
    Represents the control decision requested by the EMS.
    """
    n_modules: int      # Number of fuel cell modules to turn on
    p_batt: float       # Power requested from the battery [kW] (+ discharging, - charging)