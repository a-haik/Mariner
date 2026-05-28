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
            
            # Determine the theoretical departure time for this morning block
            target_departure = daily_start.replace(hour=morning_departure_hour, minute=0, second=0)
            
            if self.current_time < target_departure:
                # Calculate exactly how early the simulation environment was started
                lead_time_mins = (target_departure - self.current_time).total_seconds() / 60.0
                first_leg = self.schedule[0]
                
                if lead_time_mins > 5.0:
                    # Overnight transition buffer: use idling up until the final 5-minute pre-boarding window
                    idle_before_load = lead_time_mins - 5.0
                    self._record_state("idling", f"Initial Standby at {first_leg['start']}", idle_before_load)
                    self._record_state("port_operations", f"Initial Boarding at {first_leg['start']}", 5.0)
                else:
                    self._record_state("port_operations", f"Initial Boarding at {first_leg['start']}", lead_time_mins)
                
                self.current_time = target_departure
            else:
                first_leg = self.schedule[0]
                current_sec = int((self.current_time - target_departure).total_seconds())
                current_w = weather_array[current_sec] if (0 <= current_sec < len(weather_array)) else 0.0
                port_delay = self._sample_delay('port', current_w, enable_delays)
                init_dwell_duration = first_leg['dwell_mins'] + port_delay
                self._record_state("port_operations", first_leg['start'], init_dwell_duration, delay_mins=port_delay)

            # Loop sequentially through the scheduled itinerary blocks
            for idx, leg in enumerate(self.schedule):
                current_sec = int((self.current_time - self.current_time.replace(hour=morning_departure_hour, minute=0, second=0)).total_seconds())
                current_w = weather_array[current_sec] if (0 <= current_sec < len(weather_array)) else 0.0

                # 1. Outbound Maneuvering Step
                maneuver_out_delay = self._sample_delay('maneuvering', current_w, enable_delays)
                maneuver_out_dur = avg_maneuver_mins + maneuver_out_delay
                self._record_state("maneuvering", f"Leaving {leg['start']}", maneuver_out_dur, delay_mins=maneuver_out_delay)

                # 2. Inbound Maneuvering Calculation
                maneuver_in_delay = self._sample_delay('maneuvering', current_w, enable_delays)
                maneuver_in_dur = avg_maneuver_mins + maneuver_in_delay

                # 3. Open Sea Transit Step
                transit_delay = self._sample_delay('transit', current_w, enable_delays)
                net_transit_duration = leg['transit_mins'] - (2.0 * avg_maneuver_mins) + transit_delay
                net_transit_duration = max(2.0, net_transit_duration)
                self._record_state("transit", f"{leg['start']} -> {leg['end']}", net_transit_duration, delay_mins=transit_delay)

                # 4. Record Pre-Sampled Inbound Maneuvering Step
                self._record_state("maneuvering", f"Approaching {leg['end']}", maneuver_in_dur, delay_mins=maneuver_in_delay)

                # --- REFACTORED ARRIVAL PORT DWELL LOGIC ---
                port_delay = self._sample_delay('port', current_w, enable_delays)
                is_last_leg = (idx == len(self.schedule) - 1)
                
                if is_last_leg:
                    # Final trip of the day: mandatory unloading window
                    unload_mins = 5.0
                    self._record_state("port_operations", f"Final Unloading at {leg['end']}", unload_mins, delay_mins=port_delay)
                else:
                    # Standard intermediate turnaround stay (including long midday breaks)
                    total_dwell = leg['dwell_mins'] + port_delay
                    
                    # Split the dwell operationally, but keep the intermediate block as 'port_operations'
                    unload_mins = min(5.0, total_dwell / 2.0)
                    load_mins = min(5.0, total_dwell / 2.0)
                    mid_dwell_mins = max(0.0, total_dwell - (unload_mins + load_mins))

                    self._record_state("port_operations", f"Unloading at {leg['end']}", unload_mins, delay_mins=port_delay)
                    if mid_dwell_mins > 0:
                        # Replaced 'idling' with 'port_operations' to maintain full baseline auxiliary services
                        self._record_state("port_operations", f"Mid-Dwell Standby at {leg['end']}", mid_dwell_mins)
                    self._record_state("port_operations", f"Loading at {leg['end']}", load_mins)

            # 6. Night Standby Transfer Strategy
            next_day_start = datetime(daily_start.year, daily_start.month, daily_start.day, morning_departure_hour, 0) + timedelta(days=1)
            overnight_duration = (next_day_start - self.current_time).total_seconds() / 60.0
            
            if overnight_duration > 0:
                self._record_state("idling", f"Overnight at {self.schedule[-1]['end']}", overnight_duration)
            
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