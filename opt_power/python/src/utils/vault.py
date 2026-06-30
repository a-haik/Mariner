# python/src/utils/vault.py
import os
import json
import hashlib
import pandas as pd
from dataclasses import asdict

class RunVault:
    """
    Manages the deterministic hashing, caching, and retrieval of simulation runs.
    Saves scalar metrics to a lightweight JSON registry and heavy timeseries data to Parquet.
    """
    def __init__(self, base_dir: str = "../data/vault"):
        self.base_dir = base_dir
        self.telemetry_dir = os.path.join(base_dir, "telemetry")
        self.registry_path = os.path.join(base_dir, "registry.json")
        
        # Ensure directories exist
        os.makedirs(self.telemetry_dir, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)

    def _extract_primitives(self, config) -> dict:
        """
        Strips out complex objects (like Numba/NumPy arrays) from the config.
        We only hash ints, floats, strings, and booleans to guarantee determinism.
        """
        primitives = {}
        # Safely handle standard objects or dataclasses
        config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__
        
        for k, v in config_dict.items():
            if isinstance(v, (int, float, str, bool)):
                primitives[k] = v
        return primitives

    def generate_hash(self, controller_name: str, plant_name: str, train_days: list, test_day: int, is_macro: bool, config) -> str:
        """Generates a SHA-256 hash based strictly on the mathematical parameters of the run."""
        run_signature = {
            "controller": controller_name,
            "plant": plant_name,
            "train_days": sorted(train_days),
            "test_day": test_day,
            "is_macro": is_macro,
            "config": self._extract_primitives(config)
        }
        
        # sort_keys=True is critical. It guarantees the JSON string is identical every time.
        sig_string = json.dumps(run_signature, sort_keys=True)
        return hashlib.sha256(sig_string.encode('utf-8')).hexdigest()

    def get_metrics(self, hash_id: str) -> dict:
        """Returns the scalar metrics if the run exists in the registry, else None."""
        if hash_id in self.registry:
            return self.registry[hash_id]["metrics"]
        return None

    def save_run(self, hash_id: str, controller_name: str, plant_name: str, train_days: list, test_day: int, is_macro: bool, metrics: dict, telemetry_df: pd.DataFrame):
        """Writes the Parquet history to disk and registers the metrics."""
        
        # 1. Save timeseries to Parquet
        parquet_path = os.path.join(self.telemetry_dir, f"{hash_id}.parquet")
        telemetry_df.to_parquet(parquet_path, index=False)
        
        # 2. Update the JSON registry
        self.registry[hash_id] = {
            "metadata": {
                "controller": controller_name,
                "plant": plant_name,
                "train_days": train_days,
                "test_day": test_day,
                "is_macro": is_macro
            },
            "metrics": metrics,
            "file_path": parquet_path
        }
        self._save_registry()
        
    def load_telemetry(self, hash_id: str) -> pd.DataFrame:
        """Lazy loader: Fetches the massive dataframe from disk only when explicitly asked."""
        if hash_id not in self.registry:
            raise KeyError(f"Hash {hash_id} not found in registry.")
        return pd.read_parquet(self.registry[hash_id]["file_path"])