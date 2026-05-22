# MARINER Project: Ferry Modeling Synthesis

## 1. Schedule Data Analysis (Proxy: Speedrunner Jet)
We are using the Speedrunner Jet schedule as our **baseline performance metric**. Even though the target vessel is the "Super Runner Jet," this provides a real-world operational ground truth for the Rafina-Cyclades loop.

### Leg Breakdown (Normalized)
| Segment | Start Port | End Port | Transit Time | Dwell Time |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Naxos | Paros | 30m | 10m |
| 2 | Paros | Mykonos | 40m | 15m |
| 3 | Mykonos | Tinos | 20m | 10m |
| 4 | Tinos | Andros | 55m | 10m |
| 5 | Andros | Rafina | 65m | 65m (Turnaround) |
| 6 | Rafina | Andros | 75m | 10m |
| 7 | Andros | Tinos | 55m | 10m |
| 8 | Tinos | Mykonos | 20m | 10m |
| 9 | Mykonos | Paros | 45m | 10m |
| 10 | Paros | Naxos | 30m | - |

---

## 2. Mapping to Stochastic Graph Structure
We translate the schedule above into our `V = {v_port, v_approach, v_transit}` graph model.

### A. The Nodes
* **$v_{	ext{port}}$ (Dwell):** The scheduled arrival/departure delta.
    * *Implementation:* Use the "Dwell Time" column as $\mu_{	ext{port}}$.
    * *Stochasticity:* Apply $\sigma$ based on terminal congestion.
* **$v_{	ext{approach}}$ (Maneuvering):** The "hidden" time.
    * *Strategy:* Extract a fixed percentage of the transit (e.g., 10-15 mins) and treat this as a buffer. 
* **$v_{	ext{transit}}$ (Open Sea):** The "Transit Time" column.
    * *Implementation:* Treat these times as $\mu_{	ext{transit}}$. 
    * *Stochasticity:* Link $\sigma_{	ext{transit}}$ to $W(t)$ (Meso-layer weather driver).

### B. Cascading Delay Logic
If $T_{	ext{transit}} > 	ext{Planned}$, we trigger a `compression_mode`:
* Reduce $T_{	ext{port}}$ (dwell time) to recover schedule.
* If $T_{	ext{port}}$ hits a threshold (e.g., < 5 mins), signal "Failure to meet schedule."

---

## 3. Wednesday Meeting: Expert Interview Checklist
Use these questions to ground the model parameters in real experience:

1.  **Approach Buffer:** "Looking at the Rafina-Andros crossing, we have 65 minutes. How much of that is 'open sea' and how much is 'approach/maneuvering'?"
2.  **Weather Sensitivity:** "When crosswinds hit the Rafina port, does the approach duration increase linearly, or does it spike exponentially?" (Helps us choose between Linear vs. Power-Law for the physics hook).
3.  **Dwell Compression:** "In your experience, when the ferry is late, what is the 'minimum safe dwell time' for passenger/cargo handling before you are forced to skip a port or delay departure?"
4.  **Speedrunner Proxy:** "Is the Speedrunner Jet schedule realistic for our Super Runner Jet, or does the Super Runner typically handle these legs faster/slower?"

---

## 4. Immediate Next Steps
1.  **Construct the Adjacency List:** Encode the loop: `[Rafina, Approach, Andros, Approach, Tinos, Approach, Mykonos, ...]`
2.  **Define $\sigma$ (Variance):** Assign higher variance to the *Andros* and *Rafina* legs (known for challenging harbor approaches).
3.  **Simulation:** Run 10,000 iterations to determine the "Reliability Probability Distribution."