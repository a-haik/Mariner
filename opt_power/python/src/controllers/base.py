# python/src/controllers/base.py
from abc import ABC, abstractmethod
from src.core import State, Action

class ControlLaw(ABC):
    """
    Abstract base class establishing the contract layout for all power 
    distribution laws. Transitioned from full-horizon array processing 
    to discrete step-by-step evaluation for continuous ZOH simulation.
    """
    @abstractmethod
    def get_action(self, state: State) -> Action:
        """
        Evaluates the current physical state and returns the optimal control action.
        
        Parameters:
            state: The current State object (Demand, Previous Modules, SoC).
            
        Returns:
            action: The Action object (Modules to activate, Battery power split).
        """
        pass