"""
Visualization Module for Portfolio Optimization Error Analysis.
Provides comprehensive visualization tools for analyzing the impact of estimation errors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    figsize: tuple = (12, 8)
    style: str = "whitegrid"
    context: str = "notebook"
    palette: str = "viridis"
    font_scale: float = 1.2
    save_dir: Optional[Path] = None
    dpi: int = 300
    
    def __post_init__(self):
        """Set up visualization environment"""
        sns.set_style(self.style)
        sns.set_context(self.context, font_scale=self.font_scale)
        if self.save_dir:
            self.save_dir = Path(self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

class PortfolioVisualizer:
    """
    Implements visualization methods for portfolio optimization error analysis.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer with configuration"""
        self.config = config or VisualizationConfig()
        
    def _save_figure(self, fig: plt.Figure, filename: str):
        """Save figure if save_dir is configured"""
        if self.config.save_dir:
            filepath = self.config.save_dir / filename
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
    
    def plot_cel_distribution(self, results: pd.DataFrame) -> plt.Figure:
        """
        Plot Cash Equivalent Loss (CEL) distribution by error type and magnitude.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Prepare data for plotting
        plot_data = results.xs('mean', level=1, axis=1)['cel'].reset_index()
        
        # Create boxplot
        sns.boxplot(
            data=plot_data,
            x='error_magnitude',
            y='cel',
            hue='error_type',
            ax=ax
        )
        
        # Customize plot
        ax.set_title('Distribution of Cash Equivalent Loss by Error Type and Magnitude', 
                    pad=20)
        ax.set_xlabel('Error Magnitude')
        ax.set_ylabel('Cash Equivalent Loss')
        
        # Add legend with better placement
        ax.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        self._save_figure(fig, 'cel_distribution.png')
        
        return fig
    
    def plot_risk_tolerance_impact(self, results: pd.DataFrame) -> plt.Figure:
        """
        Plot impact of risk tolerance on CEL for different error types.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Prepare data
        plot_data = results.xs('mean', level=1, axis=1)['cel'].reset_index()
        
        # Create lineplot
        sns.lineplot(
            data=plot_data,
            x='risk_tolerance',
            y='cel',
            hue='error_type',
            style='error_magnitude',
            markers=True,
            dashes=False,
            ax=ax
        )
        
        # Customize plot
        ax.set_title('Impact of Risk Tolerance on Cash Equivalent Loss', 
                    pad=20)
        ax.set_xlabel('Risk Tolerance')
        ax.set_ylabel('Average Cash Equivalent Loss')
        
        # Add legend
        ax.legend(title='Error Type & Magnitude', 
                 bbox_to_anchor=(1.05, 1), 
                 loc='upper left')
        
        plt.tight_layout()
        self._save_figure(fig, 'risk_tolerance_impact.png')
        
        return fig
    
    def plot_weight_differences(self, results: pd.DataFrame) -> plt.Figure:
        """
        Plot distribution of weight differences for different error types.
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.config.figsize[0]*2, self.config.figsize[1]))
        
        # Prepare data
        max_diff_data = results.xs('mean', level=1, axis=1)['max_weight_diff'].reset_index()
        mean_diff_data = results.xs('mean', level=1, axis=1)['mean_weight_diff'].reset_index()
        
        # Plot maximum weight differences
        sns.boxplot(
            data=max_diff_data,
            x='error_magnitude',
            y='max_weight_diff',
            hue='error_type',
            ax=axes[0]
        )
        axes[0].set_title('Maximum Weight Differences')
        axes[0].set_xlabel('Error Magnitude')
        axes[0].set_ylabel('Maximum Weight Difference')
        
        # Plot mean weight differences
        sns.boxplot(
            data=mean_diff_data,
            x='error_magnitude',
            y='mean_weight_diff',
            hue='error_type',
            ax=axes[1]
        )
        axes[1].set_title('Mean Weight Differences')
        axes[1].set_xlabel('Error Magnitude')
        axes[1].set_ylabel('Mean Weight Difference')
        
        # Adjust layout
        plt.tight_layout()
        self._save_figure(fig, 'weight_differences.png')
        
        return fig
    
    def plot_active_positions(self, results: pd.DataFrame) -> plt.Figure:
        """
        Plot number of active positions for different error types.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Prepare data
        plot_data = results.xs('mean', level=1, axis=1)['active_positions'].reset_index()
        
        # Create barplot
        sns.barplot(
            data=plot_data,
            x='error_type',
            y='active_positions',
            hue='error_magnitude',
            ax=ax
        )
        
        # Customize plot
        ax.set_title('Average Number of Active Positions by Error Type', 
                    pad=20)
        ax.set_xlabel('Error Type')
        ax.set_ylabel('Average Number of Active Positions')
        
        # Add legend
        ax.legend(title='Error Magnitude', 
                 bbox_to_anchor=(1.05, 1), 
                 loc='upper left')
        
        plt.tight_layout()
        self._save_figure(fig, 'active_positions.png')
        
        return fig
    
    def plot_cel_vs_error_magnitude(self, results: pd.DataFrame) -> plt.Figure:
        """
        Plot CEL vs error magnitude (k) for different error types.
        Similar to Exhibit 4 in the paper.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Prepare data - group by error type and magnitude
        plot_data = results.xs('mean', level=1, axis=1)['cel'].groupby(
            ['error_type', 'error_magnitude']
        ).mean().reset_index()
        
        # Create line plot
        sns.lineplot(
            data=plot_data,
            x='error_magnitude',
            y='cel',
            hue='error_type',
            markers=True,
            style='error_type',
            ax=ax
        )
        
        # Customize plot
        ax.set_title('Average Cash Equivalent Loss vs Error Magnitude', pad=20)
        ax.set_xlabel('Error Magnitude (k)')
        ax.set_ylabel('Average Cash Equivalent Loss (%)')
        ax.legend(title='Parameter with Error', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        self._save_figure(fig, 'cel_vs_error_magnitude.png')
        
        return fig

    def plot_cel_ratios(self, results: pd.DataFrame) -> plt.Figure:
        """
        Plot ratios of CEL between different error types.
        Helps visualize relative importance of different error types.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Calculate mean CEL for each error type and magnitude
        means = results.xs('mean', level=1, axis=1)['cel'].groupby(
            ['error_type', 'error_magnitude', 'risk_tolerance']
        ).mean().unstack(level=0)
        
        # Calculate ratios
        ratios = pd.DataFrame({
            'means_to_variances': means['means'] / means['variances'],
            'means_to_covariances': means['means'] / means['covariances'],
            'variances_to_covariances': means['variances'] / means['covariances']
        }).reset_index()
        
        # Melt the data for plotting
        plot_data = pd.melt(
            ratios, 
            id_vars=['error_magnitude', 'risk_tolerance'],
            var_name='ratio_type',
            value_name='ratio'
        )
        
        # Create line plot
        sns.lineplot(
            data=plot_data,
            x='risk_tolerance',
            y='ratio',
            hue='ratio_type',
            style='error_magnitude',
            markers=True,
            ax=ax
        )
        
        # Customize plot
        ax.set_title('Ratios of CEL Between Error Types vs Risk Tolerance', pad=20)
        ax.set_xlabel('Risk Tolerance')
        ax.set_ylabel('CEL Ratio')
        ax.legend(title='Ratio Type', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        self._save_figure(fig, 'cel_ratios.png')
        
        return fig

    def plot_cel_statistics(self, results: pd.DataFrame) -> plt.Figure:
        """
        Plot min, mean, and max CEL for each error type.
        Similar to Exhibit 3 in the paper.
        """
        # Prepare data
        stats = results['cel'].groupby(['error_type', 'error_magnitude']).agg(['mean', 'min', 'max'])
        stats = stats.reset_index()
        
        # Create figure with subplots for each statistic
        fig, axes = plt.subplots(1, 3, figsize=(self.config.figsize[0]*2, self.config.figsize[1]))
        
        metrics = ['mean', 'min', 'max']
        titles = ['Mean CEL', 'Minimum CEL', 'Maximum CEL']
        
        for ax, metric, title in zip(axes, metrics, titles):
            sns.barplot(
                data=stats,
                x='error_magnitude',
                y=metric,
                hue='error_type',
                ax=ax
            )
            ax.set_title(title)
            ax.set_xlabel('Error Magnitude (k)')
            ax.set_ylabel('CEL (%)')
        
        plt.tight_layout()
        self._save_figure(fig, 'cel_statistics.png')
        
        return fig

    def plot_risk_tolerance_impact_detailed(self, results: pd.DataFrame) -> plt.Figure:
        """
        Plot detailed analysis of risk tolerance impact on CEL for each error type.
        """
        fig, axes = plt.subplots(1, 3, figsize=(self.config.figsize[0]*2, self.config.figsize[1]))
        
        error_types = ['means', 'variances', 'covariances']
        
        for ax, error_type in zip(axes, error_types):
            # Filter data for current error type
            type_data = results.xs('mean', level=1, axis=1)['cel'][
                results.index.get_level_values('error_type') == error_type
            ].reset_index()
            
            sns.lineplot(
                data=type_data,
                x='risk_tolerance',
                y='cel',
                hue='error_magnitude',
                markers=True,
                ax=ax
            )
            
            ax.set_title(f'Impact of Risk Tolerance\non {error_type.capitalize()} Errors')
            ax.set_xlabel('Risk Tolerance')
            ax.set_ylabel('Average CEL (%)')
            
        plt.tight_layout()
        self._save_figure(fig, 'risk_tolerance_detailed.png')
        
        return fig

    def create_paper_style_dashboard(self, results: pd.DataFrame) -> plt.Figure:
        """
        Create a comprehensive dashboard mimicking the paper's analysis style.
        """
        fig = plt.figure(figsize=(self.config.figsize[0]*2, self.config.figsize[1]*2))
        gs = fig.add_gridspec(2, 2)
        
        # CEL vs Error Magnitude (Exhibit 4 style)
        ax1 = fig.add_subplot(gs[0, 0])
        plot_data = results.xs('mean', level=1, axis=1)['cel'].groupby(
            ['error_type', 'error_magnitude']
        ).mean().reset_index()
        sns.lineplot(
            data=plot_data,
            x='error_magnitude',
            y='cel',
            hue='error_type',
            markers=True,
            ax=ax1
        )
        ax1.set_title('CEL vs Error Magnitude')
        
        # CEL Ratios
        ax2 = fig.add_subplot(gs[0, 1])
        means = results.xs('mean', level=1, axis=1)['cel'].groupby(
            ['error_type', 'risk_tolerance']
        ).mean().unstack(level=0)
        ratios = pd.DataFrame({
            'means/variances': means['means'] / means['variances'],
            'means/covariances': means['means'] / means['covariances']
        }).reset_index()
        sns.lineplot(
            data=pd.melt(ratios, id_vars=['risk_tolerance']),
            x='risk_tolerance',
            y='value',
            hue='variable',
            ax=ax2
        )
        ax2.set_title('CEL Ratios vs Risk Tolerance')
        
        # Statistics Table (Exhibit 3 style)
        ax3 = fig.add_subplot(gs[1, :])
        stats = results['cel'].groupby(['error_type', 'error_magnitude']).agg(['mean', 'min', 'max'])
        ax3.axis('tight')
        ax3.axis('off')
        table = ax3.table(
            cellText=stats.values.round(4),
            colLabels=['Mean CEL', 'Min CEL', 'Max CEL'],
            rowLabels=stats.index,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        self._save_figure(fig, 'paper_style_dashboard.png')
        
        return fig
    
    def generate_all_plots(self, results: pd.DataFrame) -> Dict[str, plt.Figure]:
        """
        Generate all available plots.
        
        Args:
            results: DataFrame with error analysis results
            
        Returns:
            Dict[str, plt.Figure]: Dictionary of generated figures
        """
        return {
            'cel_distribution': self.plot_cel_distribution(results),
            'risk_tolerance_impact': self.plot_risk_tolerance_impact(results),
            'weight_differences': self.plot_weight_differences(results),
            'active_positions': self.plot_active_positions(results),
            'cel_vs_error_magnitude': self.plot_cel_vs_error_magnitude(results),
            'cel_ratios': self.plot_cel_ratios(results),
            'cel_statistics': self.plot_cel_statistics(results),
            'risk_tolerance_detailed': self.plot_risk_tolerance_impact_detailed(results),
            'paper_style_dashboard': self.create_paper_style_dashboard(results)
        }


def create_visualizer(save_dir: Optional[Union[str, Path]] = None) -> PortfolioVisualizer:
    """
    Convenience function to create a configured visualizer.
    
    Args:
        save_dir: Optional directory to save generated plots
        
    Returns:
        PortfolioVisualizer: Configured visualizer instance
    """
    config = VisualizationConfig(save_dir=save_dir)
    return PortfolioVisualizer(config)