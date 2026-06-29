# python/src/controllers/base.py
import inspect
from abc import ABC, abstractmethod
from src.core import State, Action
from src.plants.hybrid_plant import HybridPlant

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

def build_approach(controller_cls, plant_cls=HybridPlant, solver_cls=None, is_macro=False, **kwargs):
    """
    A generic factory generator that dynamically builds the evaluation tuple for the Benchmarker.
    Uses Python introspection to automatically route the correct variables to any ControlLaw.
    """
    def factory(cfg, mc, horizon):
        # 1. Solve the offline SDP if a solver is provided
        policy_n, policy_pbatt, V = None, None, None
        raw_solution = None
        
        if solver_cls is not None:
            solver = solver_cls(cfg, mc)
            raw_solution = solver.compute_solution(horizon)
            
            # Future-Proof Unpacking: V is always the last element, policy_n is always the first.
            V = raw_solution[-1]
            policy_n = raw_solution[0]
            
            if len(raw_solution) >= 3:
                policy_pbatt = raw_solution[1]

        # 2. Create a unified context pool
        context_pool = {
            'config': cfg,
            'cfg': cfg,
            'p_grid': mc['levels'],
            'n_vals': cfg.n_vals,
            'soc_vals': getattr(cfg, 'soc_vals', None),
            'policy_matrix': policy_n,
            'policy': policy_n,
            'policy_n': policy_n,
            'policy_pbatt': policy_pbatt,
            'transition_matrix': mc['P'],
            'V_matrix': V,
            'V': V,
            'horizon_length': horizon,
            'horizon': horizon,
            'raw_solution': raw_solution  # <-- Pass the whole tuple for future augmented controllers!
        }
        
        # Merge in any custom manual overrides (like sigma, or future transient_weights)
        context_pool.update(kwargs)

        # 3. Introspection Magic: Read the target controller's __init__ parameters
        sig = inspect.signature(controller_cls.__init__)
        init_kwargs = {}
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            
            if name in context_pool:
                init_kwargs[name] = context_pool[name]
            elif param.default != inspect.Parameter.empty:
                continue # Use the default value defined in the class
            else:
                raise TypeError(f"build_approach cannot resolve required argument '{name}' for {controller_cls.__name__}")

        # 4. Instantiate and return
        ctrl = controller_cls(**init_kwargs)
        plant = plant_cls(cfg)
        
        return ctrl, plant, is_macro

    return factory