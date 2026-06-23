# python/src/plants/hybrid_plant.py
from typing import Tuple, Dict
from config import SimConfig
from src.core import State, Action
from src.plants.base import BasePlant

class HybridPlant(BasePlant):
    """
    Advanced Hybrid Plant incorporating a Li-ion Battery Pack.
    Includes physical safety clipping, Ah-throughput battery degradation, 
    and transient fuel cell penalties.
    """
    def __init__(self, config: SimConfig):
        self.config = config
        self.E_bat_max_kws = self.config.C_bat * 3600.0  # Total capacity in kWs
        self.k_trans_fc = 0.05  # [€/kW] Placeholder: Cost proportional to power delta
        self.p_fc_prev = 0.0    # Internal tracker for transient power deltas

    def step(self, state: State, action: Action, dt: float) -> Tuple[State, Dict[str, float]]:
        # 1. SAFETY OVERRIDES: Battery boundary clipping
        p_batt_req = action.p_batt
        
        # Calculate available energy bounds (kWs)
        E_avail_dis = (state.soc - self.config.soc_min) * self.E_bat_max_kws
        E_avail_chg = (self.config.soc_max - state.soc) * self.E_bat_max_kws
        
        E_req = p_batt_req * dt
        
        # Clip energy request if it violates SoC boundaries
        if E_req > E_avail_dis:      # Discharging too much
            E_actual = E_avail_dis
        elif E_req < -E_avail_chg:   # Charging too much
            E_actual = -E_avail_chg
        else:
            E_actual = E_req
            
        p_batt_actual = E_actual / dt
        soc_next = state.soc - (E_actual / self.E_bat_max_kws)
        
        # 2. Fuel Cell Load Balancing
        p_fc_total = state.P_d - p_batt_actual
        n_active = action.n_modules
        
        # 3. Fuel Cell Costs (Same electrochemical math as baseline)
        if n_active <= 0 or p_fc_total <= 0:
            c_o = 0.0 if p_fc_total <= 0 else float('inf')
        else:
            p_module = p_fc_total / n_active
            m_dot_h2 = (self.config.a0 + self.config.a1 * p_module + self.config.a2 * (p_module ** 2))
            d_fc = (1.0 / (3600.0 * self.config.tau_fc)) * (
                1.0 + self.config.alpha_fc * ((p_module - self.config.p_nom) ** 2) / (self.config.p_nom ** 2)
            )
            c_o_rate = (self.config.k_h2 * m_dot_h2 / 1000.0) + (self.config.k_fc * d_fc)
            c_o = n_active * c_o_rate * dt

        # Switching Cost
        k_s = self.config.k_fc / self.config.S_max
        c_s = k_s * abs(n_active - state.n_prev)
        
        # Transient Power Penalty (Proportional to power delta)
        c_trans = self.k_trans_fc * abs(p_fc_total - self.p_fc_prev)
        self.p_fc_prev = p_fc_total  # Update internal tracker
        
        # 4. Battery Degradation Cost (Ah-Throughput Mileage Model)
        # Assuming ~3000 equivalent full cycles (6000 half-cycles) for lifetime throughput
        lifetime_throughput_kwh = self.config.C_bat * 3000 * 2.0 
        cost_per_kwh_throughput = (self.config.c_bat_kwh * self.config.C_bat) / lifetime_throughput_kwh
        
        throughput_kwh_step = abs(p_batt_actual) * dt / 3600.0
        c_bat = cost_per_kwh_throughput * throughput_kwh_step
        
        # 5. Pack State & Telemetry
        next_state = State(P_d=0.0, n_prev=n_active, soc=soc_next)
        
        telemetry = {
            'p_fc_total': p_fc_total,
            'p_batt_actual': p_batt_actual,
            'n_active': n_active,
            'soc': soc_next,
            'cost_o': c_o,
            'cost_s': c_s,
            'cost_trans': c_trans,
            'cost_bat': c_bat,
            'cost_total': c_o + c_s + c_trans + c_bat
        }
        
        return next_state, telemetry