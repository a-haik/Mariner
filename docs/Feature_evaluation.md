# MARINER Feature Evaluation Methodology: Bridging Weak Labels and Physical Reality

## Context & The "Catch-22"
In the MARINER project, our objective is to extract highly compressed, transient-dense load profiles to design the 1,000-hour 1 MW demonstration. 

To do this, we engineered physical features (e.g., `POWER_TV_ENERGY`, `PF_DERIVATIVE`) to identify damaging regime transitions (like port maneuvering). However, our only ground-truth target is the `STATUS` column, a commercial log that is temporally lagged and physically coarse. 

If we blindly optimize a model to predict `STATUS`, we train it to mimic a flawed human observer. Instead, we use a four-pillar methodology combining weakly-supervised, unsupervised, and time-domain analyses to validate our features.

---

## 1. Statistical Divergence (Kruskal-Wallis H-Test)
**The Math:** A non-parametric ANOVA that tests whether samples originate from the same distribution. It compares the medians of a given feature across the different categorical `STATUS` regimes (e.g., Laden, Ballast, Loading).

**The Physics:** Even if the `STATUS` log is delayed by 15 minutes, the bulk statistical distribution of a feature over a 5-day steady-state 'Laden' leg should look radically different from a highly dynamic 2-day 'Maneuvering' leg.

**How to Interpret the Results:**
*   **High H-Statistic (e.g., > 100) & P-value $\approx$ 0:** The feature successfully separates the regimes. The variance of this feature is fundamentally tied to the macroscopic state of the ship.
*   **Low H-Statistic / High P-value:** The feature looks identical regardless of what the ship is doing. 
    *   *Conclusion A:* The feature is pure sensor noise.
    *   *Conclusion B:* The `STATUS` column is so inaccurate that it is washing out the physical signal. 
*   **Action:** Rank features by their H-statistic. Discard features that fail to show meaningful regime separation, as they will only add noise to a downstream HMM.

---

## 2. Feature Collinearity (Spearman Rank Correlation)
**The Math:** Spearman $\rho$ assesses monotonic relationships between variables, making it superior to Pearson for capturing non-linear shock spikes.

**The Physics:** We engineered multiple proxies for grid shocks (e.g., the exact moment a generator transitions $\Delta N_{gen}$, and the sudden drop in effective Power Factor). If these always happen simultaneously, they represent the exact same physical event.

**How to Interpret the Results:**
*   **$|\rho| > 0.80$ (Highly Correlated):** Redundancy warning. If `POWER_TV_ENERGY` and `LOAD_SYMMETRY_ERROR` are 90% correlated, they tell the same story. 
*   **Action:** Do not feed highly correlated features into the same clustering model (Curse of Dimensionality). Choose the feature that is either (A) more robust to noise, or (B) easier to justify physically to the engineering consortium designing the AST.
*   **$|\rho| \approx 0$ (Uncorrelated but High H-Stat):** Goldmine. These features capture *different* orthogonal dimensions of ship stress. For example, one captures active power demand, while the other captures reactive inductive loads.

---

## 3. Intrinsic Dimensionality (PCA Loadings)
**The Math:** Principal Component Analysis (PCA) performs an orthogonal linear transformation to project the data into a new coordinate system where the greatest variance lies on the first axes (Principal Components). *Loadings* represent the weight of each original feature in these new axes.

**The Physics:** This is a purely unsupervised check. It ignores the flawed `STATUS` column entirely and asks: "Mathematically, what is actually causing the data to fluctuate?"

**How to Interpret the Results:**
*   **Look at PC1 and PC2:** These usually capture 70-90% of the variance.
*   **High Loading Weights (e.g., > 0.5 or < -0.5):** The feature is a primary driver of the dataset's physical dynamics.
*   **Near-Zero Weights:** The feature hardly varies, or its variance is negligible compared to the rest of the system. 
*   **Action:** If a feature you mathematically engineered (like `ROT_DEG_PER_MIN`) has near-zero loadings across the first 3 PCs, it means the 5-minute sampling rate has likely washed out the signal, making it mathematically useless for clustering.

---

## 4. Transition Lead/Lag Analysis (Temporal Anticipation)
**The Math:** A point-biserial or Pearson correlation calculated between a rolling maximum of our engineered shock proxy and a boolean vector representing the exact timestamp a `STATUS` changes.

**The Physics:** This is the ultimate domain-expert check. We know the crew updates the AIS `STATUS` *after* a stressful maneuver is completed. Therefore, a true physical proxy for mechanical/electrical stress should "light up" (spike) in the data *before* the log officially changes.

**How to Interpret the Results:**
*   **High Positive Correlation:** The feature successfully anticipates the manual log change. 
    *   *Conclusion:* The engineered math is faster and more accurate than the human operator. This feature is perfect for identifying the true physical boundaries of the maneuvering regime.
*   **Zero/Negative Correlation:** The feature spikes randomly or *after* the transition. It is not a reliable leading indicator of a regime change.
*   **Action:** Use features with high lead correlation to build a threshold-based trigger. When this feature spikes, we know the ship is entering a high-stress transient zone, and we must "record" this segment of data for the 1000-hour fuel cell physical test[cite: 1].