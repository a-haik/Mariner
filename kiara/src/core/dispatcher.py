import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FerryDispatcher:
    def __init__(self, schedule, start_datetime, delay_params):
        self.schedule = schedule
        self.current_time = start_datetime
        self.delay_params = delay_params
        self.timeline = []

    def _sample_delay(self, state_type, current_weather_hazard, enable_delays=True):
        if not enable_delays:
            return 0.0
        mu = self.delay_params[state_type]['mu']
        sigma = self.delay_params[state_type]['sigma']
        if current_weather_hazard > 0.4:
            mu += self.delay_params.get('weather_penalty', 0.0)
        return np.random.lognormal(mean=mu, sigma=sigma)

    def generate_timeline(self, weather_array, days=1, morning_departure_hour=9, enable_delays=True, avg_maneuver_mins=6.0):
        self.timeline = []  # Reset historical data frame array buffer
        
        for day in range(days):
            daily_start = self.current_time
            
            # Initial Morning Setup: Passenger Loading Phase at Starting Harbor
            first_leg = self.schedule[0]
            current_sec = int((self.current_time - self.current_time.replace(hour=morning_departure_hour, minute=0, second=0)).total_seconds())
            current_w = weather_array[current_sec] if (0 <= current_sec < len(weather_array)) else 0.0
            
            port_delay = self._sample_delay('port', current_w, enable_delays)
            init_dwell_duration = first_leg['dwell_mins'] + port_delay
            self._record_state("port_dwell", first_leg['start'], init_dwell_duration, delay_mins=port_delay)

            # Loop sequentially through the scheduled itinerary blocks
            for idx, leg in enumerate(self.schedule):
                current_sec = int((self.current_time - self.current_time.replace(hour=morning_departure_hour, minute=0, second=0)).total_seconds())
                current_w = weather_array[current_sec] if (0 <= current_sec < len(weather_array)) else 0.0

                # 1. Outbound Maneuvering Step
                maneuver_out_delay = self._sample_delay('maneuvering', current_w, enable_delays)
                maneuver_out_dur = avg_maneuver_mins + maneuver_out_delay
                self._record_state("maneuvering", f"Leaving {leg['start']}", maneuver_out_dur, delay_mins=maneuver_out_delay)

                # 2. Inbound Maneuvering Calculation (Pre-sampled to solve for transit residual limits)
                maneuver_in_delay = self._sample_delay('maneuvering', current_w, enable_delays)
                maneuver_in_dur = avg_maneuver_mins + maneuver_in_delay

                # 3. Open Sea Transit Step (Subtractive Net Allocation Rule)
                transit_delay = self._sample_delay('transit', current_w, enable_delays)
                
                # Deduct the static physical maneuvering segments from the schedule baseline block
                net_transit_duration = leg['transit_mins'] - (2.0 * avg_maneuver_mins) + transit_delay
                
                # Safeguard constraints: ensure rough delays or long maneuvering settings do not yield zero transit intervals
                net_transit_duration = max(2.0, net_transit_duration)
                self._record_state("transit", f"{leg['start']} -> {leg['end']}", net_transit_duration, delay_mins=transit_delay)

                # 4. Record Pre-Sampled Inbound Maneuvering Step
                self._record_state("maneuvering", f"Approaching {leg['end']}", maneuver_in_dur, delay_mins=maneuver_in_delay)

                # 5. Arrival Port Dwell Loading Phase
                port_delay = self._sample_delay('port', current_w, enable_delays)
                dwell_duration = leg['dwell_mins'] + port_delay
                self._record_state("port_dwell", leg['end'], dwell_duration, delay_mins=port_delay)

            # 6. Night Standby Transfer Strategy
            next_day_start = datetime(daily_start.year, daily_start.month, daily_start.day, morning_departure_hour, 0) + timedelta(days=1)
            overnight_duration = (next_day_start - self.current_time).total_seconds() / 60.0
            
            if overnight_duration > 0:
                self._record_state("overnight_dwell", self.schedule[-1]['end'], overnight_duration)
            
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