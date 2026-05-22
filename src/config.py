# src/config.py
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass(frozen=True)
class PhysicalConstants:
    """Immutable physical constants for the MARINER simulation."""
    LHV_H2_KWH_KG: float = 33.32        # Lower Heating Value of Hydrogen in kWh/kg
    ETA_UPPER: float = 0.55             # Baseline / Best Case Efficiency
    ETA_LOWER: float = 0.45             # Degraded / Worst Case Efficiency
    EPSILON: float = 1e-6               # Small constant to prevent division by zero
    SPEED_THRESHOLD_KNOTS: float = 1.0  # Threshold to distinguish between 'In Port' and 'At Sea'
    P_BASE_MODULE_KW: float = 200.0     # Normalized to a single 200kW building block
    FATIGUE_EXPONENT_K: float = 2.0     # Exponent for Palmgren-Miner damage accumulation

@dataclass(frozen=True)
class FilterPresets:
    """Presets for digital signal processing (representing battery hybrid dampening)."""
    SAVGOL_DEFAULT: Dict[str, Any] = field(default_factory=lambda: {'window': 10, 'polyorder': 2})
    BUTTER_DEFAULT: Dict[str, Any] = field(default_factory=lambda: {'order': 2, 'cutoff': 0.10})

@dataclass(frozen=True)
class ColorPalette:
    """Standardized color hex codes for operational modes."""
    status_colors: Dict[str, str] = field(default_factory=lambda: {
        # Original Statuses
        'laden': '#55A868',           # Sea Green
        'ballast': '#81D8D0',         # Light Teal
        'loading': '#DD8452',         # Orange
        'discharging': '#D65F5F',     # Red
        'unloading': '#D65F5F',       # Red
        'idle': '#EAEAEA',            # Very Light Grey
        
        # New Regimes
        'sea_transit_laden': '#55A868',       
        'sea_transit_ballast': '#81D8D0',     
        'sea_loitering': "#2653CC",   # Muted Blue (High volatility at sea)
        'port_loading': '#DD8452',    
        'port_unloading': '#D65F5F',  
        'port_idle': '#EAEAEA',       
    })

# Instantiate globally accessible config objects
PHYSICS = PhysicalConstants()
FILTERS = FilterPresets()
COLOR = ColorPalette()