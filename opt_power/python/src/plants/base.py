# python/src/plants/base.py
from abc import ABC, abstractmethod
from typing import Tuple, Dict
from src.core import State, Action

class BasePlant(ABC):
    """
    Abstract base class establishing the contract for all physical hardware models.
    """
    @abstractmethod
    def step(self, state: State, action: Action, dt: float) -> Tuple[State, Dict[str, float]]:
        """
        Simulates the physical consequences of an action over time dt.
        
        Parameters:
            state: The current State object.
            action: The requested Action object.
            dt: The time duration to hold this action [s].
            
        Returns:
            Tuple containing:
            - next_state: A new State object updated with physical consequences (e.g., new SoC).
            - telemetry: A dictionary mapping string keys to metric floats 
                         (e.g., {'cost_fc': 0.5, 'cost_batt': 0.1, 'p_fc_actual': 80.0})
        """
        pass