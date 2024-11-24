# visualizer.py
"""Results visualization module."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from IPython.display import display, Markdown
from config import ExperimentConfig

class ResultVisualizer:
    """Visualization tools for sensitivity analysis results."""
    
    def __init__(self, results: Dict[int, pd.DataFrame], 
                 config: ExperimentConfig):
        self.results = results
        self.config = config
    
    def plot_distributions(self, figsize: tuple = (18, 6)) -> None:
        """Plot CEL distributions across risk tolerances."""
        fig, axes = plt.subplots(1, len(self.config.risk_tolerances), 
                                figsize=figsize)
        
        if len(self.config.risk_tolerances) == 1:
            axes = [axes]
            
        fig.suptitle('CEL Distribution by Error Type and Risk Tolerance',
                    fontsize=12, y=1.05)
        
        for i, rt in enumerate(self.config.risk_tolerances):
            data = self.results[rt]
            sns.boxplot(data=data, ax=axes[i])
            axes[i].set_title(f'Risk Tolerance = {rt}', pad=10)
            axes[i].set_ylabel('CEL (%)' if i == 0 else '')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_stats(self) -> None:
        """Generate and display summary statistics."""
        display(Markdown("## CEL Summary Statistics"))
        
        for rt in self.config.risk_tolerances:
            display(Markdown(f"### Risk Tolerance = {rt}"))
            stats = self.results[rt].describe()
            display(stats.style.format("{:.4f}"))
            
            means = self.results[rt].mean()
            ratios = pd.DataFrame({
                "Ratio": [
                    means['means']/means['variances'],
                    means['means']/means['covariances'],
                    means['variances']/means['covariances']
                ]
            }, index=['Means/Variances', 'Means/Covariances', 'Variances/Covariances'])
            
            display(Markdown("#### Importance Ratios"))
            display(ratios.style.format("{:.4f}"))
            display(Markdown("---"))

