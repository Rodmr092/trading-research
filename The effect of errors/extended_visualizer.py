# extended_visualizer.py
"""Extended visualization module for additional analyses."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from IPython.display import display, Markdown

class ExtendedVisualizer:
    """Enhanced visualization tools for extended sensitivity analysis."""
    
    def plot_error_sensitivity(self, results: pd.DataFrame, 
                             figsize: tuple = (15, 10)) -> None:
        """Plot sensitivity analysis across different error sizes."""
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        for i, metric in enumerate(['cel_means', 'cel_vars', 'cel_covs']):
            sns.boxplot(
                data=results,
                x='error_size',
                y=metric,
                hue='risk_tolerance',
                ax=axes[i]
            )
            axes[i].set_title(f'Impact of Error Size on {metric.split("_")[1].title()} CEL')
            axes[i].set_xlabel('Error Size')
            axes[i].set_ylabel('CEL (%)')
            
        plt.tight_layout()
        plt.show()
    
    def plot_size_impact(self, results: pd.DataFrame,
                        figsize: tuple = (15, 10)) -> None:
        """Plot impact of portfolio size on sensitivity."""
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        for i, metric in enumerate(['cel_means', 'cel_vars', 'cel_covs']):
            sns.boxplot(
                data=results,
                x='n_assets',
                y=metric,
                hue='risk_tolerance',
                ax=axes[i]
            )
            axes[i].set_title(f'Impact of Portfolio Size on {metric.split("_")[1].title()} CEL')
            axes[i].set_xlabel('Number of Assets')
            axes[i].set_ylabel('CEL (%)')
            
        plt.tight_layout()
        plt.show()
    
    def plot_stability_analysis(self, results: pd.DataFrame,
                              figsize: tuple = (12, 8)) -> None:
        """Plot stability analysis metrics."""
        metrics = ['mean_deviation', 'max_deviation', 'weight_std', 'turnover_ratio']
        
        fig, ax = plt.subplots(figsize=figsize)
        results_melted = results.melt(
            id_vars=['risk_tolerance'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Value'
        )
        
        sns.barplot(
            data=results_melted,
            x='Metric',
            y='Value',
            hue='risk_tolerance',
            ax=ax
        )
        ax.set_title('Portfolio Stability Metrics')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

