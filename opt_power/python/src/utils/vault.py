# python/src/utils/vault.py
import os
import json
import hashlib
import numpy as np
from dataclasses import asdict

class ModelVault:
    """
    Manages the deterministic hashing, caching, and retrieval of trained models.
    Saves the heavy Markov Transition Matrices and multi-dimensional SDP Bellman arrays.
    """
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.markov_dir = os.path.join(base_dir, "markov_models")
        self.sdp_dir = os.path.join(base_dir, "sdp_models")
        self.registry_path = os.path.join(base_dir, "model_registry.json")
        
        # Ensure subdirectories exist
        os.makedirs(self.markov_dir, exist_ok=True)
        os.makedirs(self.sdp_dir, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"markov": {}, "sdp": {}}

    def _save_registry(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)

    def _extract_math_primitives(self, config) -> dict:
        """
        Strips out complex objects (Numba/NumPy arrays) AND environment-specific
        paths to guarantee cross-machine determinism.
        """
        primitives = {}
        config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__
        
        # We must exclude paths, otherwise the hash changes depending on the computer!
        exclude_keys = {'data_dir', 'vault_dir'}
        
        for k, v in config_dict.items():
            if k in exclude_keys:
                continue
            if isinstance(v, (int, float, str, bool)):
                primitives[k] = v
        return primitives

    # =========================================================================
    # 1. MARKOV CHAIN MODEL MANAGEMENT
    # =========================================================================

    def generate_markov_hash(self, train_days: list, config) -> str:
        """The Markov Chain only depends on the training data and grid resolution parameters."""
        run_signature = {
            "train_days": sorted(train_days),
            "Ts": config.Ts,
            "N_Pd": config.N_Pd,
            "alpha_mc": config.alpha_mc
        }
        sig_string = json.dumps(run_signature, sort_keys=True)
        return hashlib.sha256(sig_string.encode('utf-8')).hexdigest()

    def save_markov_model(self, hash_id: str, mc_model: dict, train_days: list):
        """Saves the Markov matrix and state levels."""
        npz_path = os.path.join(self.markov_dir, f"{hash_id}.npz")
        
        # Unpack the dictionary and save as named arrays
        np.savez_compressed(npz_path, **mc_model)
        
        self.registry["markov"][hash_id] = {
            "train_days": sorted(train_days),
            "file_path": npz_path
        }
        self._save_registry()

    def load_markov_model(self, hash_id: str) -> dict:
        if hash_id not in self.registry["markov"]:
            return None
            
        npz_path = self.registry["markov"][hash_id]["file_path"]
        if not os.path.exists(npz_path):
            return None
            
        with np.load(npz_path) as data:
            # BUG FIX: .copy() forces the arrays into RAM before the file closes!
            return {key: data[key].copy() for key in data.files}

    # =========================================================================
    # 2. SDP BELLMAN MODEL MANAGEMENT
    # =========================================================================

    def generate_sdp_hash(self, markov_hash: str, solver_name: str, horizon_length: int, config) -> str:
        """The SDP model heavily depends on the specific Markov Chain, solver type, and math config."""
        run_signature = {
            "markov_hash": markov_hash,
            "solver": solver_name,
            "horizon_length": horizon_length,
            "config": self._extract_math_primitives(config)
        }
        sig_string = json.dumps(run_signature, sort_keys=True)
        return hashlib.sha256(sig_string.encode('utf-8')).hexdigest()

    def save_sdp_model(self, hash_id: str, markov_hash: str, solver_name: str, raw_solution: tuple):
        """Saves the Bellman matrices. raw_solution is a tuple of (policy, V) or (policy_n, policy_pbatt, V)."""
        npz_path = os.path.join(self.sdp_dir, f"{hash_id}.npz")
        
        # We unpack the tuple dynamically so it supports both baseline (2 arrays) and hybrid (3 arrays)
        np.savez_compressed(npz_path, *raw_solution)
        
        self.registry["sdp"][hash_id] = {
            "markov_hash": markov_hash,
            "solver": solver_name,
            "file_path": npz_path
        }
        self._save_registry()

    def load_sdp_model(self, hash_id: str) -> tuple:
        if hash_id not in self.registry["sdp"]:
            return None
            
        npz_path = self.registry["sdp"][hash_id]["file_path"]
        if not os.path.exists(npz_path):
            return None
            
        with np.load(npz_path) as data:
            # BUG FIX: .copy() prevents lazy-loading corruption for heavy matrices
            return tuple(data[f"arr_{i}"].copy() for i in range(len(data.files)))