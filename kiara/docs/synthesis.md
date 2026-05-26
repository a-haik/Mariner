# MARINER: Stochastic Digital Twin for High-Speed Ferry Operations

This repository contains the mathematical framework and implementation for a stochastic digital twin of a high-speed ferry operating on the Rafina-Cyclades route. The model utilizes a three-layer architecture to bridge the gap between long-term climatic trends and millisecond-level engine power dynamics.

## 1. Modeling Architecture

The model is partitioned into three distinct layers to ensure physical accuracy and computational efficiency:

* **Macro Layer (Topology & Timing):** A discrete state-machine based on the deterministic ferry schedule, enhanced with Lognormal stochastic delays to represent real-world port and transit congestion.
* **Meso Layer (Environmental Driver):** A bounded Jacobi stochastic process that models regional weather severity ($W_t \in [0, 1]$). It accounts for seasonal variations and spatial exposure using pre-calibrated intensity multipliers ($K_i$).
* **Micro Layer (Power Generation):** A bivariate system using Ornstein-Uhlenbeck (OU) processes to model engine power. This captures the physical rotational inertia of diesel engines and the rapid response of electric bow thrusters, while overlaying stochastic "gust" noise and wave-resistance penalties.

## 2. Locked Parameters (The Physics Foundation)

These parameters are derived from historical climate data (ERA5) and mechanical engineering principles. They are kept constant to ensure the digital twin remains grounded in objective reality.

| Parameter | Meaning | Derivation Basis |
| :--- | :--- | :--- |
| Jacobi $(\mu, \theta, \sigma)$ | Global weather hazard dynamics | 30-year Copernicus/ERA5 Aegean dataset |
| Spatial $K_i$ | Localized weather intensity for specific ports/straits | Geographic cross-referencing with local wind stress |
| $\tau_{diesel}$ (15s) | Rotational inertia for main engines | Marine diesel spool-up time constants |
| $\tau_{electric}$ (1s) | Response time for electric bow thrusters | Electric motor torque response physics |
| $\tau_{gust}$ (600s) | Relaxation time for high-frequency wind gusts | Meteorological standard for local wind turbulence |

## 3. Tunable Operational Parameters

These parameters represent operational "human-in-the-loop" variables. We tune these during expert validation sessions to align the model with the actual behavior of the vessel and its crew.

### Configuration & Interrogation Guide
When presenting the dashboard, use the following operational questions to extract the correct values:

| Parameter | Operational Question for the Expert |
| :--- | :--- |
| **Initial Weather ($W_0$)** | "If we simulate a worst-case scenario, what does the sea state look like at departure?" |
| **Wave Resistance (%)** | "How much additional power is required to maintain schedule speed in heavy swells?" |
| **Main/Aux Noise** | "How much do the baseline hotel/main engine loads bounce around in calm seas?" |
| **Thruster Thresholds** | "At what wind severity do captains manually engage the bow thrusters?" |
| **Maneuvering Time** | "What is the average time spent in the approach phase before docking?" |
| **Port/Transit Delays** | "What is the most frequent delay in Mykonos due to port congestion?" |
| **Weather Delay Penalty** | "How many extra minutes are added to a transit when the Meltemi is at full force?" |

## 4. Implementation Details

### The Weather Engine: Jacobi SDE
The Meso layer solves the following SDE using the Euler-Maruyama numerical scheme to ensure the weather hazard remains bounded within $[0, 1]$ without artificial clipping:

$$dW_t = \theta (\mu - W_t)dt + \sigma \sqrt{W_t(1 - W_t)} dZ_t$$

### The Power Engine: OU Inertia
Both diesel and electric streams pass through a first-order low-pass filter (the discrete-time equivalent of an Ornstein-Uhlenbeck process):

$$P_{t} = \alpha P_{t-1} + (1 - \alpha) P_{target} + \sigma \epsilon_t$$

where $\alpha = e^{-\Delta t / \tau}$. This ensures that engine power transitions are physically smooth and limited by the vessel's mechanical inertia.

## 5. Testing & Validation

To test the implementation, run the `02_expert_dashboard.ipynb` notebook. This provides an interactive control panel where you can toggle seasons, adjust physical parameters, and visually verify the power traces against the nominal baselines and stochastic delay indicators.