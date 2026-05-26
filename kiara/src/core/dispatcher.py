import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FerryDispatcher:
    def __init__(self, schedule, start_datetime, delay_params):
        """
        Initializes the state machine for the ferry topology.
        """
        self.schedule = schedule
        self.current_time = start_datetime
        self.delay_params = delay_params
        self.timeline = []

    def _sample_delay(self, state_type, current_weather_hazard):
        """Samples delay. Adds a penalty if weather is severe (W > 0.4)."""
        mu = self.delay_params[state_type]['mu']
        sigma = self.delay_params[state_type]['sigma']
        
        # Add the weather penalty if conditions are bad
        if current_weather_hazard > 0.4:
            mu += self.delay_params.get('weather_penalty', 0.0)
            
        return np.random.lognormal(mean=mu, sigma=sigma)

    def generate_timeline(self, weather_array, days=1, morning_departure_hour=7):
        """
        Generates the state machine timeline over a specified number of days.
        """
        for day in range(days):
            daily_start = self.current_time
            for leg in self.schedule:
                
                current_sec = int((self.current_time - self.current_time.replace(hour=7, minute=0, second=0)).total_seconds())
                current_w = weather_array[current_sec] if current_sec < len(weather_array) else 0.0

                # 1. Transit State
                transit_delay = self._sample_delay('transit', current_w)
                transit_duration = leg['transit_mins'] + transit_delay
                # FIXED: Now passing the delay into the record function
                self._record_state("transit", f"{leg['start']} -> {leg['end']}", transit_duration, delay_mins=transit_delay)

                # 2. Maneuvering State (Port Approach)
                if leg['dwell_mins'] > 0:
                    maneuver_delay = self._sample_delay('maneuvering', current_w)
                    maneuver_duration = self.delay_params['maneuvering']['avg_mins'] + maneuver_delay
                    # FIXED: Passing the maneuver_delay
                    self._record_state("maneuvering", f"Approaching {leg['end']}", maneuver_duration, delay_mins=maneuver_delay)

                    # 3. Port Dwell State
                    port_delay = self._sample_delay('port', current_w)
                    dwell_duration = leg['dwell_mins'] + port_delay
                    # FIXED: Passing the port_delay
                    self._record_state("port_dwell", leg['end'], dwell_duration, delay_mins=port_delay)

            # 4. Overnight Dwell State
            next_day_start = datetime(
                daily_start.year, 
                daily_start.month, 
                daily_start.day, 
                morning_departure_hour, 
                0
            ) + timedelta(days=1)
            
            overnight_duration = (next_day_start - self.current_time).total_seconds() / 60.0
            
            if overnight_duration > 0:
                self._record_state(
                    state="overnight_dwell",
                    location=self.schedule[-1]['end'],
                    duration_mins=overnight_duration
                )
            else:
                print(f"Warning: Day {day+1} operations overran the overnight threshold!")
            
            self.current_time = next_day_start
            
        return pd.DataFrame(self.timeline)
    
    def _record_state(self, state, location, duration_mins, delay_mins=0.0):
        end_time = self.current_time + timedelta(minutes=duration_mins)
        self.timeline.append({
            'state': state,
            'location': location,
            'start_time': self.current_time,
            'end_time': end_time,
            'duration_mins': round(duration_mins, 2),
            'delay_mins': round(delay_mins, 2)
        })
        self.current_time = end_time

# ==========================================
# Example Usage & Testing
# ==========================================
if __name__ == "__main__":
    cyclades_loop = [
        {'start': 'Rafina',  'end': 'Andros',  'transit_mins': 75, 'dwell_mins': 10},
        {'start': 'Andros',  'end': 'Tinos',   'transit_mins': 55, 'dwell_mins': 10},
        {'start': 'Tinos',   'end': 'Mykonos', 'transit_mins': 20, 'dwell_mins': 10},
        {'start': 'Mykonos', 'end': 'Paros',   'transit_mins': 45, 'dwell_mins': 10},
        {'start': 'Paros',   'end': 'Naxos',   'transit_mins': 30, 'dwell_mins': 10},
        {'start': 'Naxos',   'end': 'Paros',   'transit_mins': 30, 'dwell_mins': 10},
        {'start': 'Paros',   'end': 'Mykonos', 'transit_mins': 40, 'dwell_mins': 15},
        {'start': 'Mykonos', 'end': 'Tinos',   'transit_mins': 20, 'dwell_mins': 10},
        {'start': 'Tinos',   'end': 'Andros',  'transit_mins': 55, 'dwell_mins': 10},
        {'start': 'Andros',  'end': 'Rafina',  'transit_mins': 65, 'dwell_mins': 0}
    ]

    delay_parameters = {
        'transit': {'mu': 0.5, 'sigma': 0.5},
        'maneuvering': {'avg_mins': 5.0, 'mu': 0.0, 'sigma': 0.5}, # Added maneuvering params for the test
        'port':    {'mu': 1.0, 'sigma': 0.8},
        'weather_penalty': 0.5
    }

    start_dt = datetime(2026, 8, 3, 7, 0)
    
    dispatcher = FerryDispatcher(
        schedule=cyclades_loop, 
        start_datetime=start_dt, 
        delay_params=delay_parameters
    )

    # FIXED: Added a dummy weather array so the test block runs without throwing a TypeError
    dummy_weather = np.zeros(86400 * 3) 
    df_timeline = dispatcher.generate_timeline(weather_array=dummy_weather, days=3, morning_departure_hour=7)
    
    # Print specific columns so you can visually verify the delays are recording
    print(df_timeline[['state', 'location', 'duration_mins', 'delay_mins']].head(15).to_string())