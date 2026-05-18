import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_value
from sklearn.cluster import KMeans
from typing import List

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseRegimeClassifier(ABC):
    """
    Abstract Base Class for MARINER Regime Classification.
    Enforces a strict API for different mathematical HMM implementations.
    """
    def __init__(self, n_components: int = 8, inertia_weight: float = 100.0, **kwargs):
        self.n_components = n_components
        self.inertia_weight = inertia_weight
        self.is_fitted = False
        self.model_params = {}
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def _scale_continuous(self, X_cont: np.ndarray, fit: bool = False) -> np.ndarray:
        """Internal standardization utility."""
        if fit:
            self.scaler_mean_ = np.mean(X_cont, axis=0)
            self.scaler_std_ = np.std(X_cont, axis=0)
            # Add tiny epsilon to prevent division by zero for constant features
            self.scaler_std_[self.scaler_std_ == 0] = 1e-6 
            
        if self.scaler_mean_ is None or self.scaler_std_ is None:
            raise ValueError("Scaler has not been fitted.")
            
        return (X_cont - self.scaler_mean_) / self.scaler_std_

    @abstractmethod
    def fit(self, X_cont: np.ndarray, X_disc: np.ndarray):
        """
        Fits the transition and emission matrices to the data.
        X_cont: Continuous features (e.g., AE_POWER, POWER_TV_ENERGY)
        X_disc: Discrete features (e.g., NUM_GENERATORS)
        """
        pass

    @abstractmethod
    def predict(self, X_cont: np.ndarray, X_disc: np.ndarray) -> np.ndarray:
        """
        Executes Viterbi decoding to find the most likely contiguous state sequence.
        Returns an array of shape (T,) with integer state labels.
        """
        pass

    def extract_representative_profile(self, labels: np.ndarray) -> pd.DataFrame:
        """
        Utility function to be implemented later for WP9 extraction.
        Slices the raw telemetry based on stable continuous blocks.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extraction.")
        pass

class BayesianHMMClassifier(BaseRegimeClassifier):
    """
    Bayesian HMM using NumPyro. 
    Injects mechanical inertia via a heavy-diagonal Dirichlet prior.
    """
    def __init__(self, n_components: int = 8, inertia_weight: float = 100.0, seed: int = 42):
        super().__init__(n_components, inertia_weight)
        self.rng_key = jax.random.PRNGKey(seed)
        self.guide = None
        self.svi_result = None
        self.mcmc_samples = None

    def _hmm_model(self, X_cont: jnp.ndarray, X_disc: jnp.ndarray):
        K = self.n_components
        T, n_cont_features = X_cont.shape
        max_generators = 5 

        # The Inertia Prior
        alpha_prior = jnp.ones((K, K)) + jnp.eye(K) * self.inertia_weight

        with numpyro.plate("states", K):
            transition_matrix = numpyro.sample("A", dist.Dirichlet(alpha_prior))
            
            # Broad Priors (K-Means warm-start will guide SVI, but HMC relies on these)
            mu = numpyro.sample("mu", dist.Normal(0, 3).expand([n_cont_features]).to_event(1))
            sigma = numpyro.sample("sigma", dist.HalfNormal(2).expand([n_cont_features]).to_event(1))
            gen_probs = numpyro.sample("gen_probs", dist.Dirichlet(jnp.ones(max_generators)))

        def transition_fn(prev_log_prob, t):
            log_prob_cont = dist.Normal(mu, sigma).log_prob(X_cont[t]).sum(axis=-1)
            log_prob_disc = dist.Categorical(gen_probs).log_prob(X_disc[t])
            
            unnorm_log_prob = jax.scipy.special.logsumexp(
                prev_log_prob[:, None] + jnp.log(transition_matrix) + (log_prob_cont + log_prob_disc)[None, :],
                axis=0
            )
            return unnorm_log_prob, unnorm_log_prob

        init_log_prob = jnp.log(jnp.ones(K) / K)
        _, log_probs = jax.lax.scan(transition_fn, init_log_prob, jnp.arange(T))
        numpyro.factor("log_likelihood", jax.scipy.special.logsumexp(log_probs[-1]))

    def fit(self, X_cont: np.ndarray, X_disc: np.ndarray, method: str = 'svi', **kwargs):
        """
        Fits the HMM.
        method='svi' -> Fast Optimization with KMeans warm-start.
        method='hmc' -> Rigorous Hamiltonian Monte Carlo simulation.
        """
        X_cont_scaled = self._scale_continuous(X_cont, fit=True)
        X_cont_jnp = jnp.array(X_cont_scaled)
        X_disc_jnp = jnp.array(X_disc, dtype=jnp.int32)

        if method == 'svi':
            num_steps = kwargs.get('num_steps', 3000)
            
            # --- THE FIX: K-Means Warm Start to prevent Mode Collapse ---
            print(f"Warm-starting SVI means with KMeans (K={self.n_components})...")
            kmeans = KMeans(n_clusters=self.n_components, random_state=42, n_init=10)
            kmeans.fit(X_cont_scaled)
            init_values = {'mu': jnp.array(kmeans.cluster_centers_)}
            
            # Force the guide to initialize at the K-Means centroids
            self.guide = AutoDelta(self._hmm_model, init_loc_fn=init_to_value(values=init_values))
            
            optimizer = numpyro.optim.Adam(step_size=0.01)
            svi = SVI(self._hmm_model, self.guide, optimizer, loss=Trace_ELBO())
            
            print("Running SVI...")
            self.svi_result = svi.run(self.rng_key, num_steps, X_cont=X_cont_jnp, X_disc=X_disc_jnp)
            self.model_params = self.guide.median(self.svi_result.params)
            
        elif method == 'hmc':
            num_warmup = kwargs.get('num_warmup', 500)
            num_samples = kwargs.get('num_samples', 1000)
            
            print(f"Running Hamiltonian Monte Carlo (Warmup: {num_warmup}, Samples: {num_samples})...")
            # NUTS dynamically tunes the step size and trajectory of the HMC "marble"
            kernel = NUTS(self._hmm_model)
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
            
            mcmc.run(self.rng_key, X_cont=X_cont_jnp, X_disc=X_disc_jnp)
            self.mcmc_samples = mcmc.get_samples()
            
            # To interface with our Viterbi algorithm, we collapse the HMC posterior
            # back to a point estimate by taking the mean of the samples
            self.model_params = {k: jnp.mean(v, axis=0) for k, v in self.mcmc_samples.items()}
            
        else:
            raise ValueError("Method must be 'svi' or 'hmc'")

        self.is_fitted = True
        return self

    def predict(self, X_cont: np.ndarray, X_disc: np.ndarray) -> np.ndarray:
        """
        Implements the Viterbi Algorithm using the learned MAP parameters to 
        decode the optimal latent state path Z.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        X_cont_scaled = self._scale_continuous(X_cont, fit=False)
            
        A = self.model_params['A']
        mu = self.model_params['mu']
        sigma = self.model_params['sigma']
        gen_probs = self.model_params['gen_probs']
        
        T = X_cont.shape[0]
        K = self.n_components
        
        # Initialize Viterbi tracking matrices
        viterbi_log_probs = np.zeros((T, K))
        backpointers = np.zeros((T, K), dtype=int)
        
        # Step 1: Compute all emission log-probabilities upfront
        emission_log_probs = np.zeros((T, K))
        for k in range(K):
            # Scipy stats can be used here since we are out of the JAX optimization loop, 
            # but for consistency we calculate manually or via jax evaluation
            cont_logpdf = -0.5 * np.sum(np.log(2 * np.pi * sigma[k]**2) + ((X_cont_scaled - mu[k]) / sigma[k])**2, axis=1)
            disc_logpdf = np.log(gen_probs[k, X_disc] + 1e-10)
            emission_log_probs[:, k] = cont_logpdf + disc_logpdf
            
        # Step 2: Initialization at t=0
        viterbi_log_probs[0, :] = np.log(1.0 / K) + emission_log_probs[0, :]
        
        # Step 3: Forward Pass
        log_A = np.log(A + 1e-10)
        for t in range(1, T):
            for k in range(K):
                # P(Z_t = k | Z_{t-1} = j)
                trans_probs = viterbi_log_probs[t-1, :] + log_A[:, k]
                best_prev_state = np.argmax(trans_probs)
                
                backpointers[t, k] = best_prev_state
                viterbi_log_probs[t, k] = trans_probs[best_prev_state] + emission_log_probs[t, k]
                
        # Step 4: Backtracking
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(viterbi_log_probs[-1, :])
        
        for t in range(T-2, -1, -1):
            best_path[t] = backpointers[t+1, best_path[t+1]]
            
        return best_path
    
    def generate_diagnostics(self, X_cont: np.ndarray, labels: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """
        Generates a physical analysis of the learned HMM parameters and sequence.
        Inverse-transforms scaled parameters back to physical units and calculates
        state-specific run lengths.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted.")

        print("\n" + "="*70)
        print(f"MARINER HMM DIAGNOSTICS (K={self.n_components})")
        print("="*70)

        # 1. Global Temporal Dynamics
        transitions = np.sum(labels[1:] != labels[:-1])
        global_mean_dwell = (len(labels) / max(transitions, 1)) * (5 / 60)
        
        print(f"Total Sequence Length : {len(labels)} ticks ({len(labels)*5/60:.1f} hrs)")
        print(f"Total Transitions     : {transitions}")
        print(f"Global Mean Dwell     : {global_mean_dwell:.2f} hours\n")

        # 2. Run-Length Encoding for State-Specific Dwell Times
        # Identifies the start indices of every contiguous block
        changes = np.concatenate([[True], labels[1:] != labels[:-1], [True]])
        run_indices = np.where(changes)[0]
        run_lengths = np.diff(run_indices)
        run_states = labels[run_indices[:-1]]

        # 3. Extract and Inverse Transform Means and StdDevs
        scaled_mu = np.array(self.model_params['mu'])
        scaled_sigma = np.array(self.model_params['sigma'])
        
        raw_mu = (scaled_mu * self.scaler_std_) + self.scaler_mean_
        raw_sigma = scaled_sigma * self.scaler_std_ # StdDev scales multiplicatively
        
        gen_probs = np.array(self.model_params['gen_probs'])
        likely_gens = np.argmax(gen_probs, axis=1)

        # 4. Build the State Characteristic DataFrame
        state_data = []
        for k in range(self.n_components):
            # Calculate state-specific dwell time (in hours)
            state_runs = run_lengths[run_states == k]
            mean_state_dwell = np.mean(state_runs) * (5 / 60) if len(state_runs) > 0 else 0
            
            state_dict = {
                "State": k, 
                "Usage_%": f"{(np.sum(labels == k) / len(labels))*100:.1f}%",
                "Mean_Dwell(h)": round(mean_state_dwell, 2),
                "Likely_Gens": likely_gens[k]
            }
            
            # Format Mean ± StdDev for physical insight
            for i, name in enumerate(feature_names):
                state_dict[name] = f"{raw_mu[k, i]:.1f} ± {raw_sigma[k, i]:.1f}"
                
            state_data.append(state_dict)

        df_states = pd.DataFrame(state_data)
        
        print("--- LEARNED PHYSICAL EMISSIONS & DWELL TIMES (Raw Units) ---")
        print(df_states.to_markdown(index=False))

        # 5. The Transition Matrix
        print("\n--- TRANSITION PROBABILITY MATRIX (A) ---")
        A = np.array(self.model_params['A'])
        df_A = pd.DataFrame(A).round(4)
        df_A.index.name = "From State"
        df_A.columns = [f"To {c}" for c in df_A.columns]
        print(df_A.to_markdown())

        return df_states
    
    def plot_loss(self):
        """Plots the ELBO loss curve for SVI optimization."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted.")
        if self.svi_result is None:
            print("Loss plotting is only available for the SVI optimization method.")
            return

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(self.svi_result.losses, color='#2C3E50', linewidth=2)
        plt.title(f"SVI ELBO Loss Convergence (K={self.n_components}, Inertia={self.inertia_weight})")
        plt.xlabel("Iteration")
        plt.ylabel("Loss (Negative ELBO)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()