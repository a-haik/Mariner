# Calibration Methodology: Stochastic Weather Model

This document outlines the procedure to derive parameters for a Jacobi process ($W_t$) representing climate hazards, calibrated for a specific month using spatial points (ports and route midpoints).

## 1. Data Preparation

### A. Define Spatial Points
Define the set of locations for the route (Rafina-Andros-Tinos-Mykonos):
* **Nodes ($N$):** Each port (4 points) + each midpoint between ports (3 points) = **7 locations**.
* **Reference Node ($N_{ref}$):** Select one central location (e.g., the midpoint of the entire route) to serve as the master series for the Global Jacobi Process.

### B. Extract and Process Raw Data
1.  **Download:** For each location, extract hourly `Surface downward x stress` ($\tau_x$) and `Surface downward y stress` ($\tau_y$) for the target month over a historical period (e.g., last 5-10 years).
2.  **Calculate Scalar Magnitude:** Compute the scalar momentum flux to ensure direction independence:
    $$\|\tau\| = \sqrt{\tau_x^2 + \tau_y^2}$$
3.  **Normalize to $[0, 1]$:** Create a dimensionless weather variable $W$ for each location $i$. 
    *Important:* Use the global minimum and maximum across *all* locations and *all* years to ensure consistent scaling:
    $$W_{i,t} = \frac{\|\tau\|_{i,t} - \min(\|\tau\|_{all})}{\max(\|\tau\|_{all}) - \min(\|\tau\|_{all})}$$

## 2. Calibrating the Global Jacobi Process
The Global Jacobi process $W_{global}$ (based on the Reference Node) defines the regional climate "engine":
$$dW_t = \theta(\mu - W_t)dt + \sigma \sqrt{W_t(1-W_t)} dW_t$$

To achieve robustness, calculate parameters on a **Year-by-Year basis** and average them:

1.  **Annual Extraction:** For every year $y$, extract the series $W_{global, y}$.
2.  **Parameter Estimation per Year:**
    *   **Mean ($\mu_y$):** $\mu_y = \text{mean}(W_{global, y})$
    *   **Speed of Reversion ($\theta_y$):** Calculate the autocorrelation function (ACF) lag $\tau_{lag}$ where autocorrelation $\approx 0.37$. Then $\theta_y = 1 / \tau_{lag}$.
    *   **Volatility ($\sigma_y$):** $\sigma_y = \sqrt{\frac{2 \cdot \theta_y \cdot Var(W_{global, y})}{\mu_y(1-\mu_y)}}$
3.  **Final Parameters:**
    $$\mu = \text{avg}(\mu_y), \quad \theta = \text{avg}(\theta_y), \quad \sigma = \text{avg}(\sigma_y)$$

## 3. Calibrating Spatial Intensity Factors ($K$)
The Global process drives the state, but we apply fixed scaling factors ($K_i$) to account for local exposure (e.g., wind funneling).

1.  **Calculate Local Means:** For each location $i$, calculate the mean $\mu_i$ of the normalized data $W_{i,t}$ across all available years.
2.  **Determine Factor $K_i$:** Define the factor as the ratio between the local mean and the global mean:
    $$K_i = \frac{\mu_i}{\mu_{global}}$$

*   **Interpretation:**
    *   $K_i > 1$: Local conditions are naturally more intense than the regional average.
    *   $K_i < 1$: Local conditions are sheltered compared to the regional average.

## 4. Summary of Simulation Algorithm
To generate the weather hazard for any location $i$ at time $t$:

1.  **Solve the Global SDE:** 
    Use the Euler-Maruyama method to solve the Jacobi SDE for the regional "engine" $W_{global, t}$:
    $$W_{global, t+dt} = W_{global,t} + \theta(\mu - W_{global,t})dt + \sigma \sqrt{W_{global,t}(1-W_{global,t})} \cdot \epsilon \sqrt{dt}$$
    *(where $\epsilon \sim \mathcal{N}(0,1)$).*

2.  **Apply Local Intensity:** 
    Compute the specific weather hazard for location $i$:
    $$W_{i,t} = W_{global,t} \cdot K_i$$

*Note: If $W_{i,t}$ exceeds 1 due to the $K_i$ multiplier, cap the value at 1.*