# python/src/controllers/base.py
from abc import ABC, abstractmethod
import numpy as np

class ControlLaw(ABC):
    """
    Abstract base class establishing the contract layout for all power 
    distribution laws. Mirrors ControlLaw.m.
    """
    @abstractmethod
    def compute(self, P_d: np.ndarray, n0: int) -> np.ndarray:
        """
        Calculates the active module tracking array across a load timeline.
        
        Parameters:
            P_d: 1D array representing the continuous power demand profile.
            n0: Scalar integer initialization specifying initial active modules.
            
        Returns:
            n: 1D integer array containing the module allocation choices.
        """
        pass