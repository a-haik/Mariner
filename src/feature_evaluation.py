import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    
    def _get_dynamic_pcs(self, pca_df: pd.DataFrame, variance_threshold: float = 0.95) -> List[str]:
        """
        Dynamically selects the number of Principal Components needed to 
        reach the target cumulative explained variance.
        """
        if pca_df.empty or 'Explained_Variance' not in pca_df.index:
            return []
            
        explained_var = pca_df.loc['Explained_Variance']
        cumulative_var = explained_var.cumsum()
        
        # Count how many PCs we need to hit the threshold (add 1 because index is 0-based)
        num_pcs = (cumulative_var < variance_threshold).sum() + 1
        
        # Ensure we always show at least 2 PCs for 2D visualizations, 
        # and don't exceed the total available PCs
        num_pcs = max(2, min(num_pcs, len(explained_var)))
        
        return [f'PC{i}' for i in range(1, num_pcs + 1)]

    def generate_markdown_report(self, results: Dict) -> str:
        """
        Formats the evaluation results into a clean Markdown string 
        optimized for copy-pasting.
        """
        report = ["# MARINER Feature Evaluation Report\n"]
        
        if 'kruskal_wallis' in results and not results['kruskal_wallis'].empty:
            report.append("## 1. Statistical Divergence (Kruskal-Wallis H-Test)")
            report.append("*(Higher H-Statistic indicates stronger median separation across STATUS regimes)*")
            report.append(results['kruskal_wallis'].round(3).to_markdown(index=False))
            report.append("\n")
            
        if 'spearman_corr' in results and not results['spearman_corr'].empty:
            report.append("## 2. Feature Collinearity (Spearman Rank Correlation)")
            report.append("*(Values > 0.75 or < -0.75 indicate highly redundant physical information)*")
            corr = results['spearman_corr']
            upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = upper_tri.unstack().dropna().reset_index()
            high_corr.columns = ['Feature 1', 'Feature 2', 'Spearman Rho']
            high_corr = high_corr[high_corr['Spearman Rho'].abs() > 0.5].sort_values(by='Spearman Rho', key=abs, ascending=False)
            report.append(high_corr.to_markdown(index=False) if not high_corr.empty else "No significant correlations found (>0.5).")
            report.append("\n")
            
        if 'pca_loadings' in results and not results['pca_loadings'].empty:
            report.append("## 3. PCA Loadings (Intrinsic Dimensionality)")
            target_var = 0.95
            cols_to_show = self._get_dynamic_pcs(results['pca_loadings'], target_var)
            report.append(f"*(Dynamically showing PCs that explain {target_var*100:.0f}% of cumulative variance)*")
            report.append(results['pca_loadings'][cols_to_show].round(3).to_markdown())
            report.append("\n")
            
        if 'transition_lead' in results and not results['transition_lead'].empty:
            report.append("## 4. Transition Lead/Lag Analysis (Temporal)")
            report.append("*(Correlation between a feature's rolling maximum and the actual STATUS change event. Higher means it physically anticipates the manual log.)*")
            report.append(results['transition_lead'].round(3).to_markdown(index=False))
            report.append("\n")
            
        return "\n".join(report)

    def plot_diagnostics(self, results: Dict) -> None:
        """
        Generates visual representations of the feature space for local analysis in VSC.
        """
        # Determine how many valid plots we have to size the figure correctly
        valid_keys = [k for k in ['kruskal_wallis', 'spearman_corr', 'pca_loadings', 'transition_lead'] 
                      if k in results and not results[k].empty]
        
        plots = len(valid_keys)
        if plots == 0:
            print("No supported visual results found.")
            return

        cols = 2
        rows = (plots + 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        if plots > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        plot_idx = 0
        
        # 1. Plot Kruskal-Wallis H-Statistics
        if 'kruskal_wallis' in valid_keys:
            sns.barplot(data=results['kruskal_wallis'], x='H-Statistic', y='Feature', ax=axes[plot_idx], hue='Feature', legend=False, palette='viridis')
            axes[plot_idx].set_title("Kruskal-Wallis H-Statistic\n(Higher = Better Regime Separation)")
            axes[plot_idx].set_xlabel("H-Statistic")
            plot_idx += 1
        
        # 2. Plot Feature Correlations
        if 'spearman_corr' in valid_keys:
            sns.heatmap(results['spearman_corr'], annot=True, cmap='coolwarm', 
                        vmin=-1, vmax=1, center=0, fmt=".2f", ax=axes[plot_idx], 
                        cbar_kws={'label': 'Spearman Rho'}, square=True)
            axes[plot_idx].set_title("Feature Collinearity (Spearman Rank)")
            plot_idx += 1
            
        # 3. Plot PCA Loadings Heatmap
        if 'pca_loadings' in valid_keys:
            cols_to_show = self._get_dynamic_pcs(results['pca_loadings'], variance_threshold=0.95)
            loadings_only = results['pca_loadings'].drop('Explained_Variance', errors='ignore')
            
            sns.heatmap(loadings_only[cols_to_show], annot=True, cmap='PRGn', 
                        center=0, fmt=".2f", ax=axes[plot_idx], cbar_kws={'label': 'Loading Weight'}, square=True)
            axes[plot_idx].set_title("PCA Feature Loadings (95% Variance)")
            plot_idx += 1
            
        # 4. Plot Transition Lead Correlations
        if 'transition_lead' in valid_keys:
            sns.barplot(data=results['transition_lead'], x='Pre-Transition_Correlation', y='Feature', ax=axes[plot_idx], hue='Feature', legend=False, palette='magma')
            axes[plot_idx].set_title("Transition Anticipation\n(Correlation w/ Imminent Status Change)")
            axes[plot_idx].set_xlabel("Point-Biserial Correlation")
            axes[plot_idx].set_xlim(-1, 1)
            plot_idx += 1
            
        # Hide any unused subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.show()

    def _calc_statistical_divergence(self, features: List[str]) -> pd.DataFrame:
        """Applies Kruskal-Wallis H-test to evaluate regime separation."""
        regimes = self.labeled_df[self.target_col].unique()
        stats_list = []

        for feature in features:
            samples = [self.labeled_df[self.labeled_df[self.target_col] == r][feature].dropna().values for r in regimes]
            if all(len(s) > 10 for s in samples):
                h_stat, p_val = stats.kruskal(*samples)
                stats_list.append({'Feature': feature, 'H-Statistic': h_stat, 'P-Value': p_val})
        
        return pd.DataFrame(stats_list).sort_values(by='H-Statistic', ascending=False)

    def _calc_collinearity(self, features: List[str]) -> pd.DataFrame:
        """Calculates Spearman rank correlation to identify redundant features."""
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
        
        loadings = pd.DataFrame(
            pca.components_.T, 
            columns=[f'PC{i+1}' for i in range(len(features))], 
            index=features
        )
        loadings.loc['Explained_Variance'] = pca.explained_variance_ratio_
        return loadings

    def _calc_transition_lead(self, features: List[str], window: int) -> pd.DataFrame:
        """Analyzes if a feature spikes *before* the manual STATUS changes."""
        status_changed = (self.labeled_df[self.target_col] != self.labeled_df[self.target_col].shift(1))
        status_changed = status_changed & self.labeled_df[self.target_col].notna()
        
        lead_scores = []
        for feature in features:
            rolling_feat = self.df[feature].rolling(window=window, center=False).max()
            corr = rolling_feat.corr(status_changed.astype(float))
            lead_scores.append({'Feature': feature, 'Pre-Transition_Correlation': corr})
            
        return pd.DataFrame(lead_scores).sort_values(by='Pre-Transition_Correlation', ascending=False)