import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats

class VesselRegimeClassifier:
    """
    Classifies maritime operational regimes using a Gaussian Mixture Model (GMM)
    with physical priors, followed by a temporal mode filter.
    """
    
    def __init__(self, ela_priors: pd.DataFrame):
        """
        Initializes the classifier with the Electrical Load Analysis (ELA) priors.
        
        Args:
            ela_priors: A DataFrame containing 'AE_POWER(kW)' and 'NUM_GENERATORS' 
                        for the 8 expected physical regimes.
        """
        self.n_components = len(ela_priors)
        
        # Extract the physical priors to seed the Expectation-Maximization
        self.means_init = ela_priors[['TOTAL_LOAD_KW', 'NUM_GENERATORS']].values
        
        # Initialize the GMM
        # We use full covariance matrices to allow elliptical cluster shapes
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            means_init=self.means_init,
            max_iter=100,
            random_state=42 # For reproducibility during your research
        )
        
        self.is_fitted = False

    def fit_predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fits the GMM to the telemetry data and predicts the raw spatial labels.
        
        Args:
            X: DataFrame with columns matching the priors (e.g., Power and Generators).
        Returns:
            An array of raw integer labels [0 to 7].
        """
        # The EM algorithm runs here, adjusting your ELA priors to the actual data
        raw_labels = self.gmm.fit_predict(X.values)
        self.is_fitted = True
        return raw_labels

    def apply_temporal_filter(self, raw_labels: pd.Series, window_size: int = 6) -> pd.Series:
        """
        Applies a rolling majority-vote (mode) filter to enforce temporal continuity.
        
        Args:
            raw_labels: A pandas Series of the raw GMM labels.
            window_size: Number of timesteps for the rolling window. 
                         (e.g., 3 steps @ 5 mins/step = 15-minute inertia).
        Returns:
            A pandas Series of smoothed labels.
        """
        # We use a centered rolling window so the regime change aligns accurately in time
        # Lambda function applies scipy's mode to find the most frequent label in the window
        smoothed_labels = raw_labels.rolling(window=window_size, center=True).apply(
            lambda x: stats.mode(x, keepdims=False)[0], 
            raw=True
        )
        
        # Fill NaNs at the edges of the time-series with the raw labels
        return smoothed_labels.fillna(raw_labels).astype(int)

    def extract_transients(self, raw_labels: pd.Series, smoothed_labels: pd.Series) -> pd.Series:
        """
        Identifies transient events where the instantaneous physics (raw) 
        deviates from the operational intent (smoothed).
        """
        is_transient = raw_labels != smoothed_labels
        return is_transient
    
    def generate_validation_report(self, df: pd.DataFrame, raw_status_col: str = 'STATUS', 
                                   predicted_regime_col: str = 'REGIME_NAME') -> pd.DataFrame:
        """
        Performs a cross-tabulation analysis to compare raw human-logged status 
        against the GMM-predicted physical regimes.
        """
        if raw_status_col not in df.columns or predicted_regime_col not in df.columns:
            print(f"Error: Required columns {raw_status_col} or {predicted_regime_col} missing.")
            return pd.DataFrame()

        # Normalize casing for the raw status to avoid 'Idle' vs 'idle' issues
        temp_df = df.copy()
        temp_df[raw_status_col] = temp_df[raw_status_col].str.lower()

        # Create the contingency table
        mapping_matrix = pd.crosstab(
            temp_df[raw_status_col], 
            temp_df[predicted_regime_col], 
            margins=True,
            margins_name="TOTAL"
        )

        print("\n" + "="*30)
        print("MARINER REGIME VALIDATION REPORT")
        print("="*30)
        print(f"Comparison: {raw_status_col} (Raw) vs. {predicted_regime_col} (Physical)")
        print("-" * 30)
        print(mapping_matrix)
        
        # Physical insight: Identify most 'ambiguous' raw status
        # (Status with the most entropy in its physical mapping)
        print("\n--- Critical Physical Insights ---")
        for status in temp_df[raw_status_col].unique():
            if pd.isna(status): 
                continue
            counts = temp_df[temp_df[raw_status_col] == status][predicted_regime_col].value_counts()
            if len(counts) > 1:
                top_regime = counts.index[0]
                pct = (counts.iloc[0] / counts.sum()) * 100
                print(f"Raw Status '{status.upper()}' is {pct:.1f}% '{top_regime}' physically.")
        
        return mapping_matrix