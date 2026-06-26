# diagnostics.py
import unittest
import numpy as np
from src.config import SimConfig
from src.core import State, Action
from src.plants.fc_only_plant import FuelCellOnlyPlant
from src.solvers.sdp_baseline import BaselineSDPSolver

class TestMathIntegration(unittest.TestCase):
    def setUp(self):
        self.config = SimConfig()
        self.plant = FuelCellOnlyPlant(self.config)

    def test_fuel_cell_continuous_cost_math(self, test_power):
        """
        Manually verifies the electrochemical integration for 1 module at nominal 80kW load.
        """
        # 1. Setup the physical state
        state = State(P_d=80.0, n_prev=1, soc=0.7)
        action = Action(n_modules=1, p_batt=0.0)
        dt = 1.0 # Evaluate for 1 second

        # 2. Run the Plant's internal simulation
        _, telemetry = self.plant.step(state, action, dt)

        # 3. Independent Manual Calculation (The "Whiteboard" Proof)
        p_module = 80.0
        # Expected H2 flow [g/s]
        expected_m_dot = self.config.a0 + (self.config.a1 * p_module) + (self.config.a2 * (p_module**2))
        # Expected Degradation [1/s] (Should be baseline since p_module == p_nom)
        expected_d_fc = 1.0 / (3600.0 * self.config.tau_fc)
        # Expected Total Cost [€/s]
        expected_cost = (self.config.k_h2 * expected_m_dot / 1000.0) + (self.config.k_fc * expected_d_fc)

        # 4. Assert correctness up to 6 decimal places
        self.assertAlmostEqual(telemetry['cost_o'], expected_cost, places=6, 
                               msg="Plant continuous operating cost does not match the electrochemical formulas.")

class TestSafetyInvariants(unittest.TestCase):
    def setUp(self):
        self.config = SimConfig()
        self.plant = FuelCellOnlyPlant(self.config)

    def test_division_by_zero_prevention(self):
        """
        Proves the plant safely penalizes actions that request 0 active modules during active demand.
        """
        state = State(P_d=100.0, n_prev=1, soc=0.7)
        action = Action(n_modules=0, p_batt=0.0)
        
        _, telemetry = self.plant.step(state, action, dt=1.0)
        self.assertEqual(telemetry['cost_o'], float('inf'), 
                         "Plant failed to apply infinite penalty for turning off all modules under load.")

class TestToyBellmanSolver(unittest.TestCase):
    def test_toy_dp_matrix_logic(self):
        """
        Builds a trivially small 2x2 state space to ensure the Bellman recursion 
        selects the obvious optimal path without getting confused by indices.
        """
        # Override config to force a strict ceiling
        toy_config = SimConfig(p_max=100.0, p_nom=80.0, n_vals=np.array([1, 2]), n0=1)
        
        # 2 Demand States: 80kW (Low), 160kW (High)
        toy_mc = {
            'levels': np.array([80.0, 160.0]),
            'P': np.array([[1.0, 0.0], [0.0, 1.0]]) # Completely predictable transition
        }
        
        solver = BaselineSDPSolver(toy_config, toy_mc)
        policy = solver.compute_policy_matrix(horizon_length=2)
        
        # Scenario: It is Time 0. Demand is 160kW (Index 1). 
        # We currently have 1 module active (Index 0).
        # We MUST switch to 2 modules (Action Index 1) because 1 module at 160kW exceeds p_max (100kW).
        optimal_action_idx = policy[0, 1, 0]
        
        self.assertEqual(optimal_action_idx, 1, 
                         "DP Solver failed to switch to 2 modules when load exceeded p_max!")

if __name__ == '__main__':
    unittest.main(verbosity=2)