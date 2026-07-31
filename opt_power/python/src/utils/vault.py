# python/src/utils/vault.py
import os
import json
import hashlib
import numpy as np

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
        registry = {"markov": {}, "sdp": {}}
        
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        registry.update(data)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[Vault] WARNING: Registry file corrupted or unreadable ({e}). Falling back to empty registry.")
        
        # Guarantee required dictionary keys exist
        registry.setdefault("markov", {})
        registry.setdefault("sdp", {})
        return registry

    def _save_registry(self):
        # Write atomically using a temporary file to prevent corruption on sudden interruptions
        dir_name = os.path.dirname(self.registry_path) or "."
        temp_path = os.path.join(dir_name, f".{os.path.basename(self.registry_path)}.tmp")
        
        with open(temp_path, 'w') as f:
            json.dump(self.registry, f, indent=4)
            
        os.replace(temp_path, self.registry_path)

    def _extract_math_primitives(self, config) -> dict:
        """
        Strips out complex objects (Numba/NumPy arrays) to guarantee cross-machine determinism.
        Tracks NumPy array shapes so grid resolution changes trigger recomputes.
        """
        primitives = {}
        config_dict = {}
        
        # 1. Catch formally defined dataclass fields
        if hasattr(config, '__dataclass_fields__'):
            import dataclasses
            config_dict.update(dataclasses.asdict(config))
            
        # 2. Catch dynamically assigned attributes from __post_init__
        if hasattr(config, '__dict__'):
            config_dict.update(config.__dict__)
            
        for k, v in config_dict.items():
            if isinstance(v, (int, float, str, bool)):
                primitives[k] = v
            elif isinstance(v, np.number): # CRITICAL: Catch numpy scalars (np.int64, etc.)
                primitives[k] = v.item()   # Convert to native python type
            elif isinstance(v, np.ndarray):
                primitives[f"{k}_shape"] = list(v.shape) # Hash the physical shape of the grid
            elif isinstance(v, list):
                primitives[f"{k}_len"] = len(v)
                
        return primitives

    # =========================================================================
    # 1. MARKOV CHAIN MODEL MANAGEMENT
    # =========================================================================

    def generate_markov_hash(self, train_days: list, config) -> str:
        """The Markov Chain depends on training data and all grid/physics parameters."""
        run_signature = {
            "train_days": sorted(train_days),
            "config": self._extract_math_primitives(config) # Use the full primitive extractor!
        }
        sig_string = json.dumps(run_signature, sort_keys=True)
        return hashlib.sha256(sig_string.encode('utf-8')).hexdigest()

    def save_markov_model(self, hash_id: str, mc_model: dict, train_days: list, offline_time: float):
        """Saves the Markov matrix, state levels, and computation time."""
        npz_path = os.path.join(self.markov_dir, f"{hash_id}.npz")
        
        # Unpack the dictionary and save as named arrays
        np.savez_compressed(npz_path, **mc_model)
        
        self.registry["markov"][hash_id] = {
            "train_days": sorted(train_days),
            "file_path": npz_path,
            "offline_time": offline_time
        }
        self._save_registry()

    def load_markov_model(self, hash_id: str):
        if hash_id not in self.registry["markov"]:
            return None
            
        npz_path = self.registry["markov"][hash_id]["file_path"]
        if not os.path.exists(npz_path):
            return None
            
        with np.load(npz_path) as data:
            model = {key: data[key].copy() for key in data.files}
            
        offline_time = self.registry["markov"][hash_id].get("offline_time", 0.0)
        return model, offline_time

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

    def save_sdp_model(self, hash_id: str, markov_hash: str, solver_name: str, raw_solution: tuple, offline_time: float):
        """Saves the Bellman matrices and computation time."""
        npz_path = os.path.join(self.sdp_dir, f"{hash_id}.npz")
        
        np.savez_compressed(npz_path, *raw_solution)
        
        self.registry["sdp"][hash_id] = {
            "markov_hash": markov_hash,
            "solver": solver_name,
            "file_path": npz_path,
            "offline_time": offline_time
        }
        self._save_registry()

    def load_sdp_model(self, hash_id):
        npz_path = os.path.join(self.sdp_dir, f"{hash_id}.npz")
        
        if not os.path.exists(npz_path):
            return None
            
        try:
            # Attempt to open and decompress the binary file
            with np.load(npz_path) as data:
                raw_solution = tuple(data[f"arr_{i}"].copy() for i in range(len(data.files)))
                
            offline_time = self.registry["sdp"].get(hash_id, {}).get("offline_time", 0.0)
            return raw_solution, offline_time
            
        except Exception as e:
            # If zlib, EOFError, or BadZipFile throws an error due to corruption
            print(f" \n[Vault] WARNING: Corrupted cache file detected for {hash_id}.")
            print(f" -> Deleting corrupted file and forcing recomputation... (Error: {e})")
            
            # Delete the corrupted file so it doesn't break future runs
            if os.path.exists(npz_path):
                os.remove(npz_path)
                
            # Return None to trigger a standard Cache MISS in the benchmarker
            return None