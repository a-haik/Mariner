# python/src/environment.py
import numpy as np
from numba import njit

@njit(cache=True)
def simulate_markov_chain_demand(horizon_steps: int, states: np.ndarray, 
                                 transition_matrix: np.ndarray, initial_demand: float) -> np.ndarray:
    """
    Numba port of the sequential Markov Chain generation loop inside Demand.m.
    Simulates a synthetic load trajectory using state transition probabilities.
    """
    P_d = np.zeros(horizon_steps)
    
    # Find the closest discrete state index to the target initialization value
    current_state_idx = np.abs(states - initial_demand).argmin()
    P_d[0] = states[current_state_idx]
    
    for t in range(1, horizon_steps):
        probs = transition_matrix[current_state_idx, :]
        cumprobs = np.cumsum(probs)
        
        # Sample the next state transition using standard Monte Carlo inversion
        r = np.random.rand()
        
        # Fast Numba-compatible search for cumulative threshold boundary
        next_idx = 0
        for i in range(len(cumprobs)):
            if r <= cumprobs[i]:
                next_idx = i
                break
                
        current_state_idx = next_idx
        P_d[t] = states[current_state_idx]
        
    return P_d


@njit(cache=True)
def create_gaussian_random_walk_matrix(states: np.ndarray, sigma: float) -> np.ndarray:
    """
    Converts a continuous Gaussian random walk model into a discrete Markov Chain matrix.
    Perfect replication of create_gaussian_random_walk_matrix inside create_transition_matrix.m.
    """
    N_d = len(states)
    P = np.zeros((N_d, N_d))
    
    for i in range(N_d):
        current_state = states[i]
        if current_state <= 0:
            P[i, 0] = 1.0  # Boundary condition: Absorbing barrier at zero load limits
            continue
            
        for j in range(N_d):
            next_state = states[j]
            delta = next_state - current_state
            
            # Continuous Gaussian transition probability density evaluation
            P[i, j] = np.exp(-0.5 * (delta / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
            
        # Row-wise normalization to satisfy basic probability rules
        row_sum = np.sum(P[i, :])
        if row_sum > 0:
            P[i, :] = P[i, :] / row_sum
            
    return P