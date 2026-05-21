# src/config.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class PhysicalConstants:
    """Immutable physical constants."""
    LHV_H2_KWH_KG: float = 33.32  # Lower Heating Value of Hydrogen in kWh/kg
    ETA_UPPER: float = 0.55       # Baseline / Best Case Efficiency
    ETA_LOWER: float = 0.45       # Degraded / Worst Case Efficiency
    EPSILON: float = 1e-6         # Small constant to prevent division by zero
    SPEED_THRESHOLD_KNOTS: float = 1.0  # Threshold to distinguish between 'In Port' and 'At Sea'
    P_BASE_MODULE_KW: float = 200.0  # Normalized to a single 200kW building block
    FATIGUE_EXPONENT_K: float = 2.0

@dataclass(frozen=True)
class FilterPresets:
    """Presets for digital signal processing."""
    SAVGOL_DEFAULT: Dict[str, Any] = None
    BUTTER_DEFAULT: Dict[str, Any] = None

    def __post_init__(self):
        # Using object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, 'SAVGOL_DEFAULT', {'window': 10, 'polyorder': 2})
        object.__setattr__(self, 'BUTTER_DEFAULT', {'order': 2, 'cutoff': 0.05})

@dataclass(frozen=True)
class ColorPalette:
    status_colors = {
            # 1. Original Statuses
            'laden': '#55A868',       # Sea Green
            'ballast': '#81D8D0',     # Light Teal
            'loading': '#DD8452',       # Orange
            'discharging': '#D65F5F',   # Same as unloading
            'idle': '#EAEAEA',            # Very Light Grey

            # 2. New Regimes (if they appear in the data)
            'Sea_Transit_Laden': '#55A868',       # Sea Green
            'Sea_Transit_Ballast': '#81D8D0',     # Light Teal
            'Sea_Loitering': "#2653CC",       # Muted Red (High volatility at sea)
            'Port_Loading': '#DD8452',        # Orange
            'Port_Unloading': '#D65F5F',      # Purple (High power draw)
            'Port_Idle': '#EAEAEA',           # Very Light Grey
        }

# Instantiate globally accessible config objects
PHYSICS = PhysicalConstants()
FILTERS = FilterPresets()
COLOR = ColorPalette()