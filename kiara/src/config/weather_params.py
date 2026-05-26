# Dummy parameters for a "Moderate" day. 
# We will replace these with real ERA5 data shortly.
JACOBI_PARAMS = {
    'mu': 0.4,       # Baseline weather severity [0-1]
    'theta': 0.15,   # Mean reversion rate (approx 6-hour relaxation)
    'sigma': 0.05    # Volatility of the weather front
}