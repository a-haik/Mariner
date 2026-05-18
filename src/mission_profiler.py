import pandas as pd
import numpy as np

class MissionProfiler:
    """Classifies raw telemetry into physical mission phases using block-averaged kinematics."""
    
    def __init__(self, df: pd.DataFrame, speed_threshold: float = 1.0):
        self.df = df.copy()
        self.speed_threshold = speed_threshold

    def classify_phases(self) -> pd.DataFrame:
        raw_status = self.df['STATUS'].str.lower()
        
        # 1. Identify raw contiguous blocks and their mean speeds
        raw_block_id = (raw_status != raw_status.shift()).cumsum()
        block_mean_speed = self.df.groupby(raw_block_id)['SHIP SPEED(knots)'].transform('mean')
        
        # 2. Determine binary location: Sea vs. Port
        # A ship is at sea if it is explicitly transiting, OR if it is idling but moving fast (Loitering)
        is_sea = raw_status.isin(['laden', 'ballast', 'sea going']) | \
                 ((raw_status == 'idle') & (block_mean_speed > self.speed_threshold))
        
        # 3. Create a unique ID for each continuous Port Stay and Sea Voyage
        stay_id = (is_sea != is_sea.shift()).cumsum()
        
        # 4. Broadcast the dominant cargo operation across the entire Port Stay
        # If *any* row in a port stay is 'loading', the whole stay is considered a loading phase
        has_loading = self.df.groupby(stay_id)['STATUS'].transform(
            lambda x: x.str.lower().isin(['loading']).any()
        )
        has_discharge = self.df.groupby(stay_id)['STATUS'].transform(
            lambda x: x.str.lower().isin(['discharging', 'unloading']).any()
        )
        
        # 5. Apply the classification logic
        conditions = [
            # SEA PHASES
            is_sea & raw_status.isin(['laden', 'sea going']),
            is_sea & raw_status.isin(['ballast']),
            is_sea & (raw_status == 'idle'),  # Dynamic Loitering
            
            # PORT PHASES (The entire stay inherits the cargo op if it exists)
            ~is_sea & has_loading,
            ~is_sea & has_discharge,
            ~is_sea  # True Port Idle (moored, no cargo ops occur during this stay)
        ]
        
        choices = [
            'Transit_Laden',
            'Transit_Ballast',
            'Sea_Loitering',
            'Port_Loading',
            'Port_Unloading',
            'Port_Idle'
        ]
        
        self.df['PHASE'] = np.select(conditions, choices, default='Unknown')
        
        # 6. Create final unique ID for our newly defined physical blocks
        phase_changes = self.df['PHASE'] != self.df['PHASE'].shift()
        self.df['PHASE_ID'] = phase_changes.cumsum()
        
        return self.df