import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict

class FeatureEvaluator:
    """
    Evaluates engineered physical features using multiple mathematical strategies 
    to bypass weakly-supervised/lagged target labels.
    """
    def __init__(self, df: pd.DataFrame, target_col: str = 'STATUS'):
        self.df = df.copy()
        self.target_col = target_col
        # Drop rows where target is missing for supervised checks
        self.labeled_df = self.df.dropna(subset=[self.target_col])

    def evaluate(self, features: List[str], strategy: str = 'all', **kwargs) -> Dict:
        """
        Master execution method.
        Supported strategies: 'statistical', 'unsupervised', 'temporal', 'all'
        """
        results = {}
        
        if strategy in ['statistical', 'all']:
            results['kruskal_wallis'] = self._calc_statistical_divergence(features)
        
        if strategy in ['unsupervised', 'all']:
            results['spearman_corr'] = self._calc_collinearity(features)
            results['pca_loadings'] = self._calc_pca_loadings(features)
            
        if strategy in ['temporal', 'all']:
            window = kwargs.get('window_size', 12) # default 1 hr (12 * 5min)
            results['transition_lead'] = self._calc_transition_lead(features, window)
            
        return results

    def _calc_statistical_divergence(self, features: List[str]) -> pd.DataFrame:
        """Applies Kruskal-Wallis H-test to evaluate regime separation."""
        regimes = self.labeled_df[self.target_col].unique()
        stats_list = []

        for feature in features:
            # Gather arrays of the feature for each regime
            samples = [self.labeled_df[self.labeled_df[self.target_col] == r][feature].dropna().values for r in regimes]
            
            # Ensure we have enough data to run the test
            if all(len(s) > 10 for s in samples):
                h_stat, p_val = stats.kruskal(*samples)
                stats_list.append({'Feature': feature, 'H-Statistic': h_stat, 'P-Value': p_val})
        
        # Sort by highest H-statistic (best separation)
        res_df = pd.DataFrame(stats_list).sort_values(by='H-Statistic', ascending=False)
        return res_df

    def _calc_collinearity(self, features: List[str]) -> pd.DataFrame:
        """Calculates Spearman rank correlation to identify redundant features."""
        # Spearman captures non-linear monotonic relationships better than Pearson
        return self.df[features].corr(method='spearman')

    def _calc_pca_loadings(self, features: List[str]) -> pd.DataFrame:
        """Uses PCA to determine which features drive the most intrinsic dataset variance."""
        clean_subset = self.df[features].dropna()
        if clean_subset.empty:
            return pd.DataFrame()

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_subset)
        
        pca = PCA()
        pca.fit(scaled_data)
        
        # Extract the loadings (how much each feature contributes to each Principal Component)
        loadings = pd.DataFrame(
            pca.components_.T, 
            columns=[f'PC{i+1}' for i in range(len(features))], 
            index=features
        )
        # Add explained variance ratio as a reference
        loadings.loc['Explained_Variance'] = pca.explained_variance_ratio_
        return loadings

    def _calc_transition_lead(self, features: List[str], window: int) -> pd.DataFrame:
        """
        Analyzes if a feature spikes *before* the manual STATUS changes.
        This isolates the actual physical event from the lagged human logging.
        """
        # Create a boolean series where a regime transition occurs
        # Shift it back slightly to create an 'event window'
        status_changed = (self.labeled_df[self.target_col] != self.labeled_df[self.target_col].shift(1))
        status_changed = status_changed & self.labeled_df[self.target_col].notna()
        
        lead_scores = []
        for feature in features:
            # We want to see if the rolling maximum of the feature correlates highly 
            # with the transition event, proving it captures the lead-up to the change.
            rolling_feat = self.df[feature].rolling(window=window, center=False).max()
            
            # Simple correlation between the transition event and the rolling feature spike
            # (Note: point-biserial correlation is technically better here, but Pearson is a decent quick proxy)
            corr = rolling_feat.corr(status_changed.astype(float))
            lead_scores.append({'Feature': feature, 'Pre-Transition_Correlation': corr})
            
        return pd.DataFrame(lead_scores).sort_values(by='Pre-Transition_Correlation', ascending=False)