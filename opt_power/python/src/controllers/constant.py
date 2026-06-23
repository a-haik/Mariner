# python/src/controllers/constant.py
from src.controllers.base import ControlLaw
from src.core import State, Action
from config import SimConfig

class ConstantControl(ControlLaw):
    """
    Keeps the number of active modules invariant across changes in load demand.
    """
    def __init__(self, config: SimConfig):
        self.n_constant = config.n0

    def get_action(self, state: State) -> Action:
        return Action(n_modules=self.n_constant, p_batt=0.0)