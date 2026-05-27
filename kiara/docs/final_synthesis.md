# Comprehensive Technical Synthesis: KIARA Synthetic Telemetry Engine

## 1. Executive Summary & Architectural Paradigm
The KIARA Synthetic Telemetry Engine serves as a high-fidelity digital twin framework developed for the European MARINER project. Its objective is to close the gap between macroscopic historical routing models and high-frequency microscopic power system physical behaviors for the *KIARA* high-speed catamaran ferry fleet operating in the Cyclades region of the Aegean Sea. 

Architecturally, the model utilizes a **decoupled multi-scale state-space formulation**. It segregates macroscopic timetable routing logistics (the State Machine Layer) from high-frequency aerodynamic, hydrodynamic, mechanical, and electrical transient interactions (the Physical Core Layer). By grounding parameters within the statistical invariants of a 30-year Copernicus/ERA5 climatological database, the simulator moves away from heuristic assumptions, anchoring its thresholds and scaling limits in verifiable environmental physics.

---

## 2. Macroscopic Layer: State Machine & Timetable Logistics
The macroscopic topology tracks the spatial coordinates of the vessel across specific network legs ($i$) defined by a schedule matrix $\mathcal{S}$. At any continuous time $t$, the macro-state of the vessel is categorized into a discrete operational regime:
$$\text{Regime}(t) \in \{\text{Transit}, \, \text{Maneuvering}, \, \text{Loading}, \, \text{Overnight}\}$$

### Scheduling, Subtractive Constraints & Stochastic Delays
The duration of each leg or harbor block is subject to cumulative operational delay overruns driven by human routing variables and structural port congestion ($\xi_{\text{delay}}$). 

Crucially, to preserve the rigid berth-to-berth timetable limits of real-world ferry operations, the open-sea transit block is calculated as a **Subtractive Net Residual**. The time spent maneuvering out of the departure port and into the arrival port is deducted from the scheduled leg duration:

$$\Delta T_{\text{actual,maneuver}} = \Delta T_{\text{avg,maneuver}} + \xi_{\text{delay}}(W_{\text{eff}})$$
$$\Delta T_{\text{actual,transit}} = \max\Big(2.0, \, \Delta T_{\text{Schedule,Transit}} - 2 \cdot \Delta T_{\text{avg,maneuver}}\Big) + \xi_{\text{delay}}(W_{\text{eff}})$$

The stochastic delay component $\xi_{\text{delay}}$ is governed by a boolean presentation bypass flag ($b_{\text{delays}}$):

$$\xi_{\text{delay}}(W_{\text{eff}}) = \begin{cases} 
0.0 & \text{if } b_{\text{delays}} = \text{False} \\
\text{Lognormal}(\mu_{\text{regime}}, \, \sigma_{\text{regime}}) & \text{if } b_{\text{delays}} = \text{True} \text{ and } W_{\text{eff}} \le 0.4 \\
\text{Lognormal}(\mu_{\text{regime}} + \mu_{\text{penalty}}, \, \sigma_{\text{regime}}) & \text{if } b_{\text{delays}} = \text{True} \text{ and } W_{\text{eff}} > 0.4 
\end{cases}$$

Where $\mu_{\text{regime}}$ and $\sigma_{\text{regime}}$ represent state-specific parameters stored in the configuration layer, and $\mu_{\text{penalty}}$ represents an environmental schedule penalty added when the wind hazard breaches a critical boundary.

---

## 3. Environmental Framework: Multi-Scale Meteorology
The environment is modeled as a continuous hazard index $W(t) \in [0.0, 1.0]$, representing the localized aggregate force of aerodynamic wind stress and hydrodynamic sea state. The weather is resolved across three specific mathematical timescales.

#### A. Meso-Scale Baseline (Hourly Resolution)
The global wind field is driven by a bounded **Jacobi Stochastic Differential Equation (SDE)**, ensuring that environmental variables naturally mean-revert without exceeding physical boundaries:
$$dW_g = \theta_j (\mu_j - W_g)dt + \sigma_j \sqrt{W_g (1 - W_g)} dB_t$$
Where $\theta_j$ is the hourly mean-reversion rate, $\mu_j$ is the calibrated historical monthly mean wind hazard, $\sigma_j$ is the diffusion volatility factor, and $dB_t$ represents an incremental standard Brownian motion process.

#### B. Slow Atmospheric Drift (Minute Resolution)
To simulate localized atmospheric micro-climates, an additive low-frequency noise layer is superimposed and passed through a relaxation filter to emulate slow wind field shifts:
$$\tau_{\text{gust,slow}} \frac{d\eta_{\text{slow}}}{dt} + \eta_{\text{slow}} = \epsilon_{\text{slow}}(t), \quad \epsilon_{\text{slow}} \sim \mathcal{N}(0, \, \sigma_{\text{slow,gust,base}}^2)$$

#### C. Fast Localized Turbulent Wind Gusts (Second Resolution)
High-frequency turbulent wind gusts—critical for analyzing sudden aerodynamic load shifts in narrow island straits—are generated via a highly responsive mean-reverting process whose variance scales directly as a fraction of the historical monthly mean weather hazard $\mu_j$:
$$\tau_{\text{gust,fast}} \frac{d\eta_{\text{fast}}}{dt} + \eta_{\text{fast}} = \epsilon_{\text{fast}}(t), \quad \epsilon_{\text{fast}} \sim \mathcal{N}\left(0, \, (\mu_j \cdot \phi_{\text{gust,turb}})^2\right)$$
Where $\phi_{\text{gust,turb}}$ represents the user-controlled turbulence amplitude percentage.

#### D. Geographic Mapping Layer
The global wind hazard vector is mapped onto specific local coordinates using a spatial multiplier $K_i$, which scales the environmental threat level based on the distinct geographic wind-shadow profiles of individual ports and shipping lanes:
$$W_{\text{eff}}(t) = \text{clip}\Big( (W_g(t) \cdot K_i) + \eta_{\text{slow}}(t) + \eta_{\text{fast}}(t), \, 0.0, \, 1.0 \Big)$$

---

## 4. Microscopic Physical Core Layer: Coupled Scaling Laws
The physical layer processes the effective environmental index $W_{\text{eff}}(t)$ at a $1.0\text{-second}$ step resolution, generating the target command tracks for the propulsion drivetrain and auxiliary electrical infrastructure.

### 1. Statistical Operational Boundaries
The system eliminates abstract threshold variables by deriving its operational triggers straight from long-term statistical invariants:
* **Clean-Hull Reference Anchor ($W_{\text{baseline}}$)**: Fixed to the historical yearly average $\mu_{\text{annual}}$, matching the sea trial baseline where the ship achieves nominal contract speed in calm water.
* **Thruster Activation Cut-In ($W_{\text{cut,in}}$)**: Set at $1.0$ standard deviation above the annual norm, representing a routine fresh windy day:
  $$W_{\text{cut,in}} = \mu_{\text{annual}} + 1.0 \cdot \sigma_{\text{annual}}$$
* **System Maximum Saturation Limit ($W_{\text{saturation}}$)**: Set at $2.5$ standard deviations above the annual norm, representing strong, regular seasonal gales:
  $$W_{\text{saturation}} = \mu_{\text{annual}} + 2.5 \cdot \sigma_{\text{annual}}$$

### 2. Propulsion Plant Model (Main Engines)
The main engine command tracking core is dictated by a piecewise continuous response law that accounts for wave drag penalties at sea and active harbor-pinning thrust overrides during docking sequences:

$$P_{\text{main,target}}(t) = \begin{cases}
P_{\text{main,sea}} \cdot \Big(1.0 + \kappa_{\text{wave}} \cdot \max(0.0, \, W_{\text{eff}}(t) - W_{\text{baseline}})\Big) & \text{if Regime}(t) = \text{Transit} \\
P_{\text{main,maneuver}} & \text{if Regime}(t) = \text{Maneuvering and } W_{\text{eff}}(t) \le W_{\text{cut,in}} \\
P_{\text{main,maneuver}} + \gamma(t) \cdot (P_{\text{main,port,max}} - P_{\text{main,maneuver}}) & \text{if Regime}(t) = \text{Maneuvering and } W_{\text{eff}}(t) > W_{\text{cut,in}} \\
P_{\text{main,port,base}} & \text{if Regime}(t) = \text{Loading and } W_{\text{eff}}(t) \le W_{\text{cut,in}} \\
P_{\text{main,port,base}} + \gamma(t) \cdot (P_{\text{main,port,max}} - P_{\text{main,port,base}}) & \text{if Regime}(t) = \text{Loading and } W_{\text{eff}}(t) > W_{\text{cut,in}} \\
0.0 & \text{if Regime}(t) = \text{Overnight}
\end{cases}$$

Where $\kappa_{\text{wave}}$ is the added wave resistance coefficient, and $\gamma(t)$ is the continuous interpolation ramp scaling factor inside the wind corridor:
$$\gamma(t) = \text{clip}\left( \frac{W_{\text{eff}}(t) - W_{\text{cut,in}}}{W_{\text{saturation}} - W_{\text{cut,in}}}, \, 0.0, \, 1.0 \right)$$

### 3. Auxiliary Electrical Network Model
To maintain strict adherence to energy-conservation principles, the auxiliary electrical network is modeled as **two structurally isolated parallel circuits**, preventing high-frequency wind noise from unrealistically siphoning power from internal hotel systems.

* **Rigid Base Load Circuit ($P_{\text{hotel}}$)**: Represents the continuous unyielding load floor required to keep vital shipboard systems (HVAC, passenger decks, illumination) operational:
  $$P_{\text{hotel,target}}(t) = \begin{cases} 
  P_{\text{aux,sea}} & \text{if Transit} \\
  P_{\text{aux,maneuver}} & \text{if Maneuvering} \\
  P_{\text{aux,port,base}} & \text{if Loading} \\
  P_{\text{aux,hotel}} & \text{if Overnight}
  \end{cases}$$
* **Asymmetric Hydrodynamic Thruster Circuit ($P_{\text{thruster}}$)**: Represents the additional power drawn by the variable-frequency bow thruster drives to counteract crosswind hull shear. This vector is bounded at zero, ensuring it can only add load on top of the hotel base:
  $$P_{\text{thruster,target}}(t) = \begin{cases}
  0.0 & \text{if Regime}(t) \in \{\text{Transit}, \, \text{Overnight}\} \\
  0.0 & \text{if Regime}(t) \in \{\text{Maneuvering}, \, \text{Loading}\} \text{ and } W_{\text{eff}}(t) \le W_{\text{cut,in}} \\
  \gamma(t) \cdot (P_{\text{aux,spike,max}} - P_{\text{hotel,target}}(t)) & \text{if Regime}(t) \in \{\text{Maneuvering}, \, \text{Loading}\} \text{ and } W_{\text{eff}}(t) > W_{\text{cut,in}}
  \end{cases}$$

### 4. Dynamic Inertia and Smoothing Filters
To accurately replicate physical system behavior, the blocky command tracks are converted into smooth physical curves by passing them through linear first-order differential filters that reflect system mass and operator response delays:
$$\tau_{\text{diesel}} \frac{dP_{\text{main,c}}}{dt} + P_{\text{main,c}} = P_{\text{main,target}}(t)$$
$$\tau_{\text{electric}} \frac{dP_{\text{hotel,c}}}{dt} + P_{\text{hotel,c}} = P_{\text{hotel,target}}(t)$$
$$\tau_{\text{human}} \frac{dP_{\text{thruster,c}}}{dt} + P_{\text{thruster,c}} = P_{\text{thruster,target}}(t)$$

These continuous-time lags are executed in code via an infinite impulse response (IIR) filter derived through an exact discrete Z-transform mapping ($\alpha = e^{-dt/\tau}$):
$$y[n] = (1 - \alpha) \cdot x[n] + \alpha \cdot y[n-1]$$

---

## 5. Stochastic Noise, Memory & Measurement Precision
To match the characteristic "wandering" profile of authentic shipboard telemetry logs, the framework overlays an auto-correlated noise envelope combined with high-frequency measurement error.

### 1. Relative Multiplicative Ornstein-Uhlenbeck (OU) Noise Engine
Rather than using arbitrary additive white noise, the macro-environmental deviations follow mean-reverting stochastic paths. Crucially, this load wandering operates on an **external operational timescale ($\tau_{\text{drift}}$)**, representing physical wave-hull interactions and course corrections, entirely decoupled from internal engine inertia:

$$d\chi_{\text{main}} = -\frac{1}{\tau_{\text{drift}}} \chi_{\text{main}} dt + \left(\sigma_{\text{fraction}} \cdot P_{\text{main,c}}(t)\right) \cdot \sqrt{\frac{2}{\tau_{\text{drift}}}} dB_t^{\text{main}}$$

$$d\chi_{\text{hotel}} = -\frac{1}{\tau_{\text{drift}}} \chi_{\text{hotel}} dt + \left(\sigma_{\text{fraction}} \cdot P_{\text{hotel,c}}(t) \cdot \lambda_{\text{aux}}\right) \cdot \sqrt{\frac{2}{\tau_{\text{drift}}}} dB_t^{\text{hotel}}$$

$$d\chi_{\text{thruster}} = -\frac{1}{\tau_{\text{drift}}} \chi_{\text{thruster}} dt + \left(\sigma_{\text{fraction}} \cdot P_{\text{thruster,c}}(t)\right) \cdot \sqrt{\frac{2}{\tau_{\text{drift}}}} dB_t^{\text{thrust}}$$

Where $\lambda_{\text{aux}}$ is a fixed reduction multiplier reflecting the insulated nature of internal hotel buses relative to severe wave-slapping propulsion forces.

### 2. Physical Envelope Interlocking
The components are compiled into raw telemetry tracks and bounded at zero to prevent unphysical negative calculations:
$$P_{\text{main,raw}}(t) = \max\Big(0.0, \, P_{\text{main,c}}(t) + \chi_{\text{main}}(t)\Big)$$
$$P_{\text{aux,raw}}(t) = \max\Big(0.0, \, P_{\text{hotel,c}}(t) + \chi_{\text{hotel}}(t)\Big) + \max\Big(0.0, \, P_{\text{thruster,c}}(t) + \chi_{\text{thruster}}(t)\Big)$$

### 3. Instrumental Telemetry Fuzz Layer
Finally, to match the sensor tolerances of an industrial Power Management System (PMS), a symmetrical, zero-mean Gaussian distribution layer scales a measurement precision envelope ($\delta_{\text{instrument}}$) relative to the current active load:
$$\epsilon_{\text{main,fuzz}}(t) \sim \mathcal{N}\left(0, \, (\delta_{\text{instrument}} \cdot P_{\text{main,raw}}(t))^2\right)$$
$$\epsilon_{\text{aux,fuzz}}(t) \sim \mathcal{N}\left(0, \, (\delta_{\text{instrument}} \cdot P_{\text{aux,raw}}(t))^2\right)$$

The output fields recorded by the digital twin database are:
$$P_{\text{main,actual}}(t) = P_{\text{main,raw}}(t) + \epsilon_{\text{main,fuzz}}(t)$$
$$P_{\text{aux,actual}}(t) = P_{\text{aux,raw}}(t) + \epsilon_{\text{aux,fuzz}}(t)$$

---

## 6. System Parameters Reference Framework

### 1. Grounded Physical & Statistical Invariants (Locked Constants)
These variables are fixed within `vessel_specs.py`, representing the unyielding geometric, atmospheric, mechanical, and instrumental constraints of the vessel.

| Variable | Structural Value | Real-World Engineering Meaning |
| :--- | :--- | :--- |
| $W_{\text{annual,mean}}$ | $0.09142$ | Clean-hull performance baseline calculated over a 30-year Aegean ERA5 history. |
| $W_{\text{annual,std}}$ | $0.06056$ | Historical standard deviation tracking long-term weather volatility. |
| $\tau_{\text{diesel}}$ | $15.0\text{ seconds}$ | Rotational block inertia, turbocharger lag, and governor loop delay of the Ruston propulsion plant. |
| $\tau_{\text{electric}}$ | $1.0\text{ second}$ | Combined electromagnetic and inverter loop switching time constant of the auxiliary distribution network. |
| $\tau_{\text{gust,slow}}$ | $600.0\text{ seconds}$ | Meteorological relaxation window for meso-scale atmospheric field trends. |
| $\tau_{\text{gust,fast}}$ | $30.0\text{ seconds}$ | Micro-scale wind turbulence duration tracking individual wind gust boundaries. |
| $\tau_{\text{human}}$ | $8.0\text{ seconds}$ | Ergonomic lag matching a captain holding a bridge joystick during docking adjustments. |
| $\tau_{\text{drift}}$ | $180.0\text{ seconds}$ | Hydrodynamic hull-wave drift relaxation constant representing operational load wander. |
| $\alpha_{\text{thruster,start}}$| $1.0 \cdot \sigma$ | Multiplier defining the start of wind effects where crosswinds exceed routine parameters ($\approx \text{Beaufort } 5$). |
| $\beta_{\text{thruster,max}}$ | $2.0 \cdot \sigma$ | Multiplier defining gale conditions requiring max maneuvering output ($\approx \text{Beaufort } 7$). |
| $\lambda_{\text{aux}}$ | $0.25$ | Fixed dampening scalar reflecting the isolation of hotel breakers from raw hydrodynamic wave impact forces. |

### 2. Tunable Parameters & Real-World Interrogation Map
These variables are exposed directly via dashboard widgets. They serve as calibration knobs to align the dataset with a captain's operational experience during expert validation sessions.

| Slider Parameter | Code Variable Reference | Real-World Operational Analogy | Expert Interrogation Prompt |
| :--- | :--- | :--- | :--- |
| **Season** | `month_dd` | Determines the active monthly baseline ($\mu_j$, $\sigma_j$) using either peaceful spring or rough summer *Meltemi* periods. | *"Which seasonal layout are we calibrating against? Should we evaluate a calm spring transit or a rough August Meltemi run?"* |
| **Initial Weather Level (W0)** | `w0_slider` | Sets the initial environmental hazard value at departure. | *"If we simulate a worst-case scenario, what does the sea state look like when pushing off from the Rafina terminal?"* |
| **Added Wave Resistance Factor** | `wave_res` | Governs the slope of propulsion power consumption as hull resistance scales upward in rough open water. | *"When running steady cruise revolutions in heavy swells, how much extra baseline thermal load do the waterjets draw to keep schedule speed?"* |
| **Load Drift Amplitude (%)** | `sigma_frac` | Adjusts the percentage window of the slow-moving, auto-correlated power fluctuations across the entire plant. | *"Look at this steady-state section during open-sea transit. Does this natural load wander look realistic, or do the automated fuel governors smooth it out more?"* |
| **Wind Gust Turbulence (%)** | `gust_frac` | Controls the maximum peak intensity of high-frequency local wind ripples hitting the vessel. | *"In the narrow passages between Tinos and Andros, do local gusts hit the hull in sharp, aggressive bursts like this, or is the wind pressure more uniform?"* |
| **Telemetry Sensor Error (%)** | `delta_inst_w` | Calibrates the high-frequency instrument noise floor generated by measurement tolerances. | *"How clean is the raw telemetry logged by your power management shunts? Do you see a sensor fuzz envelope of 0.5% or closer to 1.5%?"* |
| **Maneuver Time (Mins)** | `maneuver_time_w` | Calibrates the fixed duration spent navigating in and out of the harbor limits. | *"We currently block out 5 minutes for maneuvering in Mykonos. Is that realistic, or does it take longer?"* |
| **Simulate Delays** | `delay_toggle` | Toggle switch to test the model's behavior under either rigid timeline targets or random logistical schedule overruns. | *(Operational control flag used to switch between isolating clean transient shapes and checking long-term timetable resilience).* |
| **Overlay 1-Min Trend** | `trend_toggle` | Toggle to overlay moving averages, making it easy to compare raw high-frequency signals with steady electrical trends. | *(Presentation control flag used to verify that auxiliary fluctuations never plunge below the physical hotel load floor).* |