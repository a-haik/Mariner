# MARINER: Data-Driven Operational Profiling & Scenario Engineering

## 1. Project Overview
The European MARINER project aims to validate and demonstrate a reliable, efficient, scalable, and low-cost 1 MW PEM Fuel Cell (PEMFC) system for maritime applications. Operating within Work Package 9 (Scale-Up & End-user Engagement), this codebase acts as the analytical bridge between historical vessel telemetry and the physical testing phase. 

By ingesting real-world operational data from the SCORPIO Handymax tankers, this framework profiles temporal load demands. The objective is to predict hydrogen ($H_2$) consumption bounds and quantify structural fatigue across various 1000-hour testing scenarios, allowing engineers to optimize the 2028 demonstration protocols against strict fuel budgets and hardware constraints.

---

## 2. Scientific Foundation & Methodological Framework

The analytical pipeline is built upon deterministic physical modeling rather than stochastic or black-box machine learning approaches. This ensures that every computed metric—from required battery capacity to membrane degradation—maintains strict conservation of energy and remains interpretable for hardware sizing.

### 2.1 Telemetry Regularization & Spatial-Temporal Synchronization
Raw vessel telemetry suffers from inconsistent sampling rates, missing entries, and manual log lagging. The `VesselDataLoader` class normalizes this by forcing the raw CSV data onto a strictly monotonic 5-minute time grid (`pd.Series.resample`). 
To preserve the physical continuity of the vessel's kinematics, linear variables (like power and speed) are interpolated linearly over time, while circular variables (like heading) are decomposed into trigonometric components:
$$H_{sin} = \sin(H_{raw}), \quad H_{cos} = \cos(H_{raw})$$
These components are averaged and reconstructed using `numpy.arctan2` to prevent artificial wraparound errors during aggregation.

### 2.2 Hybrid Energy Management & Battery Buffer Optimization
To protect the fuel cell stack from severe transient loads, the system simulates a charge-sustaining Hybrid Energy Management System (EMS). In `data_processing.py`, this is achieved by passing the raw electrical demand ($P_{AE}$) through a zero-phase digital Butterworth low-pass filter via `scipy.signal.filtfilt`. 

This effectively splits the power delivery: the fuel cell handles the low-frequency baseload ($P_{FC}$), while a theoretical battery pack absorbs the high-frequency transients. The instantaneous battery power demand is defined as:
$$P_{batt}(t) = P_{AE}(t) - P_{FC}(t)$$
By varying the filter's cutoff frequency ($f_c$), we alter the hardware trade-off space. To deduce the actual required battery capacity for a given voyage block, the algorithm centers the battery power and integrates it over time to find the maximum energy capacity excursion:
$$\Delta E_{batt} = \max \left( \int \left( P_{batt}(t) - \bar{P}_{batt} \right) dt \right) - \min \left( \int \left( P_{batt}(t) - \bar{P}_{batt} \right) dt \right)$$
The final installed capacity estimates assume a safe operational Depth of Discharge (DoD) envelope of 60%.

### 2.3 PEMFC Degradation & Rainflow Fatigue Accumulation
To evaluate the relative wear on the PEMFC membrane caused by dynamic load cycling, the `MissionProfiler` adapts structural mechanics principles to electrochemistry. Using the `rainflow` Python library, the algorithm extracts closed hysteresis loops from the time-series power demand of each operational block.

We apply the Palmgren-Miner linear damage rule to calculate a time-normalized, dimensionless Fatigue Activity Rate ($D$) per hour:
$$D = \frac{1}{T_{block}} \sum_{i} n_i \left( \frac{\Delta P_i}{P_{base}} \right)^k$$
Where $n_i$ is the cycle count at amplitude $\Delta P_i$. To ensure the metric scales correctly to the MARINER architecture, the amplitude is normalized against a $P_{base} = 200 \text{ kW}$ modular building block. The exponent $k=2.0$ acts as an intensive penalty, disproportionately weighting deep, high-amplitude power swings over minor high-frequency noise.

### 2.4 Thermodynamic Efficiency & Hydrogen Mass Flow
Predicting hydrogen mass requirements is bounded by the thermodynamic efficiency of the fuel cell stack ($\eta$), which decays over its lifecycle. The instantaneous hydrogen mass flow rate is derived directly from the filtered fuel cell demand:
$$\dot{m}_{H_2}(t) = \frac{P_{FC}(t)}{\eta \cdot LHV_{H_2}}$$
Given the Lower Heating Value ($LHV_{H_2} = 33.32 \text{ kWh/kg}$), the pipeline computes all final scenario metrics deterministically across an uncertainty envelope spanning from an optimal beginning-of-life state ($\eta_{upper} = 0.55$) to a degraded end-of-life state ($\eta_{lower} = 0.45$).

---

## 3. Architecture & Data Pipeline

The repository is structured as a functional, unidirectional data pipeline. It decouples raw I/O, human-in-the-loop labeling, signal processing, and mathematical aggregation to ensure reproducibility across different vessel datasets.

### Pipeline Execution Flow
1. **`src/preprocess_entry.py` (Standardization):**
   - **Input:** Raw telemetry files (`.csv`, `.xlsx`).
   - **Process:** Intelligently parses mixed date formats, removes legacy metadata footers, and dynamically corrects ASCII-encoded GPS coordinates to standard decimal degrees.
   - **Output:** Cleaned `.csv` files stored in `data/interim/`.

2. **`labeling_tool.py` (Interactive Human-in-the-loop UI):**
   - **Input:** Interim telemetry files.
   - **Process:** Launches a Streamlit web application. It pre-chunks the dataset using kinematic heuristics, allowing users to visually verify operational modes via Folium trajectory maps, Plotly global context maps, and high-frequency time-series charts. Users can merge/split blocks and optionally append geospatial port contexts.
   - **Output:** Verified operational block registries (`data/labeled/`) and fully mapped high-frequency telemetry (`data/processed/`).

3. **`src/data_loader.py` & `src/data_processing.py` (EMS Simulation):**
   - **Input:** Verified `processed` telemetry.
   - **Process:** Enforces a strict, monotonic 5-minute time grid using linear and circular interpolation. Applies the zero-phase Butterworth filter to isolate battery buffer requirements while safely passing the human-verified labels directly to the physics engine.

4. **`src/mission_profiler.py` & `src/visualizer.py` (Fatigue & Scenarios):**
   - **Process:** Groups the continuous voyage phases into a `Block Registry`, executes Rainflow fatigue counting, integrates $H_2$ consumption, and scales these blocks into 1000-hour analytical testing scenarios. Visualizers render the resulting multi-dimensional tradeoff spaces.

---

## 4. 1000-Hour Scenario Generation

Because the 2028 demonstration phase operates under a strictly capped hydrogen budget, the `ScenarioManager` (within `mission_profiler.py`) dynamically recompiles the historical `Block Registry` into four distinct 1000-hour testing profiles. 

Each scenario is evaluated against a **Standard** (average historical distribution) and a **Low-Cost** (bottom 25% optimized fuel flow) subset to expose OPEX vs. CAPEX degradation trade-offs.

### 4.1 The Four MARINER Profiles
1. **Historical Baseline:** A direct, proportional scaling of the vessel's empirical mode distribution to a 1000-hour window.
2. **Cold-Ironing Integration (Shore Power):** Replaces all heavy port loading/unloading demands with minimal baseline hotel loads, simulating a harbor grid connection. 
3. **Docked Shutdown (PEMFC-Off):** Simulates a complete, cold shutdown of the fuel cell stack during all port operations. The saved hours are re-normalized across active sea transit modes.
4. **Extended Transit Optimization:** Leverages a **Proportional Ratio Approach** to isolate long-haul logistics.
   - *The Math:* It filters the registry for transit blocks exceeding the historical median duration. It then computes a historical port-to-transit ratio ($\gamma = \frac{\sum T_{port}}{\sum T_{transit}}$) and applies this scalar to the long-haul hours to synthetically generate realistic, dependent port-handling times, preserving global shipping semantics.

---

## 5. Hardware Constraints & Project Assumptions

The pipeline enforces several immutable physical boundaries derived directly from the MARINER proposal (Part B) and hardware specifications:

* **Absolute Power Envelope:** The PEMFC stack is strictly capped at a **1,000 kW** maximum output. Transients exceeding this ceiling are mathematically forced onto the battery buffer.
* **Stack Modularization:** Fatigue amplitude penalties are normalized against a standard **200 kW** physical building block ($P_{base}$).
* **Efficiency Bounds:** To account for stack degradation over the project lifecycle, hydrogen flow predictions span an uncertainty envelope bounded by $\eta_{upper} = 0.55$ (Best Case) and $\eta_{lower} = 0.45$ (Degraded Case).
* **Battery Sizing Buffer:** Peak usable battery capacity excursions are divided by a **0.60** factor to respect a 60% Depth of Discharge (DoD) hardware limitation, providing realistic installed pack estimates.

---

## 6. Repository Structure

```text
scorpio/
├── .docs/                  # Supplemental documentation and design notes
├── data/
│   ├── raw/                # Raw legacy telemetry (.csv, .xlsx)
│   ├── interim/            # Standardized telemetry (post-preprocessing)
│   ├── labeled/            # Human-verified blocks & port contexts
│   └── processed/          # Final 5-min telemetry mapped with status
├── notebooks/
│   └── test.ipynb          # End-to-end integration and scenario testing
├── src/                    # Core analytical source code
│   ├── preprocess_entry.py # Data standardization entrypoint
│   ├── data_loader.py
│   ├── data_processing.py
│   ├── mission_profiler.py
│   └── visualizer.py
├── labeling_tool.py        # Interactive Streamlit UI application
├── .gitignore              # Prevents tracking of datasets
├── README.md               # Project documentation
└── requirements.txt        # Managed dependency manifest
```

---

## 7. Setup & Installation

### Prerequisites
* **Python:** Version 3.12 or higher.
* **Environment:** A virtual environment (`venv`) is strongly recommended to isolate dependencies.

### Installation
1. Clone the repository and navigate to the project root:
   ```bash
   git clone <repository_url>
   cd mariner-scorpio
2. Initialize and activate the virtual environment:
   ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3. Install the core dependencies from the manifest:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
### Executing the Pipeline
1. Ensure your raw vessel telemetry CSVs are placed in the `data/raw/` directory.

2. **Preprocess the data** to fix dates and GPS encoding:
   ```bash
   python src/preprocess_entry.py your_raw_file.csv
3. **Launch the Labeling Tool** to verify operational modes:
   ```bash
   streamlit run labeling_tool.py
4. **Run the Analysis**: Use `notebooks/01_test.ipynb` to execute the workflow.

## 8. Author & Acknowledgments
Author: Adam Haïk - Dual-Degree Engineering & Physics Student (Mines Paris - PSL / ENS - PSL)

Institution: NORCE Norwegian Research Centre AS

Note: The documentation and initial codebase scaffolding in this repository were co-authored with the assistance of an AI engineering co-pilot.