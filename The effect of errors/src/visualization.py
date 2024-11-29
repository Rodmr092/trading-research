"""
Visualization Module for Portfolio Error Analysis.
Provides comprehensive visualization tools for analyzing estimation errors
in portfolio optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioVisualizer:
    """
    Creates standardized visualizations for portfolio optimization error analysis.
    """
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualizer with consistent style settings.
        
        Args:
            style: matplotlib style to use
            figsize: default figure size
        """
        self.style = style
        self.figsize = figsize
        self.colors = {
            'means': '#2ecc71',
            'variances': '#e74c3c',
            'covariances': '#3498db'
        }
        # Configuración básica del estilo
        plt.style.use(style)
        sns.set_style("whitegrid")
        
    def setup_figure(self, title: str) -> Tuple[plt.Figure, plt.Axes]:
        """Create figure with consistent styling."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title, pad=20)
        return fig, ax
        
    def plot_cel_heatmap(self, results_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Create heatmap showing CEL variation with error magnitude and risk tolerance."""
        plt.figure(figsize=(15, 5))
        
        for i, error_type in enumerate(['means', 'variances', 'covariances'], 1):
            plt.subplot(1, 3, i)
            
            # Filtrar por tipo de error y extraer los valores CEL
            mask = results_df.index.get_level_values('error_type') == error_type
            data = results_df[mask]
            
            # Crear una matriz pivote directamente del MultiIndex
            pivot_data = data[('cel', 'mean')].unstack('error_magnitude')
            
            sns.heatmap(
                pivot_data,
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'CEL'}
            )
            
            plt.title(f'CEL Heatmap - {error_type.capitalize()}')
            plt.xlabel('Error Magnitude')
            plt.ylabel('Risk Tolerance')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
    def plot_cel_boxplots(self,
                         results_df: pd.DataFrame,
                         save_path: Optional[str] = None) -> None:
        """
        Create box plots showing CEL distribution across error types.
        
        Args:
            results_df: DataFrame with error analysis results
            save_path: Optional path to save the plot
        """
        fig, ax = self.setup_figure('CEL Distribution by Error Type and Magnitude')
        
        # Prepare data for plotting
        plot_data = results_df.reset_index()
        
        # Create box plots
        sns.boxplot(data=plot_data,
                   x='error_magnitude',
                   y=('cel', 'mean'),
                   hue='error_type',
                   palette=self.colors)
        
        plt.xlabel('Error Magnitude')
        plt.ylabel('Cash Equivalent Loss (CEL)')
        plt.legend(title='Error Type')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
    def plot_weight_differences(self, results_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Create violin plots showing distribution of weight differences."""
        fig, ax = self.setup_figure('Portfolio Weight Differences Distribution')
        
        # Preparar datos
        plot_data = results_df.reset_index()
        plot_data['mean_weight_diff'] = plot_data['mean_weight_diff'].astype(float)
        
        # Crear violin plot
        sns.violinplot(
            data=plot_data,
            x='error_magnitude',
            y='mean_weight_diff',
            hue='error_type',
            split=True,
            palette=self.colors
        )
        
        plt.xlabel('Error Magnitude')
        plt.ylabel('Mean Weight Difference')
        plt.legend(title='Error Type')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
    def plot_risk_return_scatter(self,
                               results_df: pd.DataFrame,
                               save_path: Optional[str] = None) -> None:
        """
        Create scatter plot comparing risk-return characteristics.
        
        Args:
            results_df: DataFrame with error analysis results
            save_path: Optional path to save the plot
        """
        fig, ax = self.setup_figure('Risk-Return Comparison')
        
        # Prepare data
        plot_data = results_df.reset_index()
        
        for error_type in ['means', 'variances', 'covariances']:
            mask = plot_data['error_type'] == error_type
            
            # Plot optimal portfolios
            plt.scatter(plot_data[mask]['optimal_risk'],
                       plot_data[mask]['optimal_return'],
                       label=f'{error_type} - Optimal',
                       color=self.colors[error_type],
                       alpha=0.6,
                       marker='o')
                       
            # Plot suboptimal portfolios
            plt.scatter(plot_data[mask]['suboptimal_risk'],
                       plot_data[mask]['suboptimal_return'],
                       label=f'{error_type} - Suboptimal',
                       color=self.colors[error_type],
                       alpha=0.3,
                       marker='x')
        
        plt.xlabel('Portfolio Risk (Volatility)')
        plt.ylabel('Expected Return')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
    def plot_cel_confidence_bands(self, results_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Create line plot with confidence bands showing CEL variation."""
        logger.info("Starting plot_cel_confidence_bands...")
        
        # Crear nueva figura
        plt.figure(figsize=self.figsize)
        
        try:
            for error_type in ['means', 'variances', 'covariances']:
                # Filtrar datos por tipo de error
                mask = results_df.index.get_level_values('error_type') == error_type
                data = results_df[mask]
                
                # Obtener magnitudes de error únicas y ordenadas
                error_magnitudes = np.sort(data.index.get_level_values('error_magnitude').unique())
                means = []
                stds = []
                
                # Calcular medias y std para cada magnitud de error
                for mag in error_magnitudes:
                    mag_data = data[data.index.get_level_values('error_magnitude') == mag]
                    means.append(mag_data[('cel', 'mean')].mean())
                    stds.append(mag_data[('cel', 'std')].mean())
                
                means = np.array(means)
                stds = np.array(stds)
                
                # Plotear línea principal
                plt.plot(error_magnitudes, means, 
                        label=error_type.capitalize(),
                        color=self.colors[error_type],
                        marker='o',
                        linewidth=2)
                
                # Agregar bandas de confianza
                plt.fill_between(error_magnitudes,
                            means - 1.96 * stds,
                            means + 1.96 * stds,
                            alpha=0.2,
                            color=self.colors[error_type])
            
            plt.title('CEL Variation with Error Magnitude', pad=20)
            plt.xlabel('Error Magnitude')
            plt.ylabel('Cash Equivalent Loss (CEL)')
            plt.legend(title='Error Type')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                logger.info(f"Saving confidence bands plot to {save_path}")
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                # Verificar que el archivo se haya guardado
                if not Path(save_path).exists():
                    raise RuntimeError(f"Failed to save confidence bands plot to {save_path}")
                
                logger.info("Confidence bands plot saved successfully")
                
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error in plot_cel_confidence_bands: {str(e)}")
            raise
        finally:
            plt.close('all')

    def create_analysis_dashboard(self, results_df: pd.DataFrame, output_dir: str = 'figures') -> None:
        """Generate and save all visualization plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plots_to_generate = [
            (self.plot_cel_heatmap, 'cel_heatmap.png'),
            (self.plot_cel_boxplots, 'cel_boxplots.png'),
            (self.plot_weight_differences, 'weight_differences.png'),
            (self.plot_risk_return_scatter, 'risk_return.png'),
            (self.plot_cel_confidence_bands, 'cel_confidence.png')
        ]
        
        for plot_func, filename in plots_to_generate:
            file_path = output_path / filename
            plt.close('all')  # Limpiar cualquier figura existente
            
            try:
                logger.info(f"Generating {filename}...")
                plot_func(results_df, str(file_path))
                
                # Verificar que el archivo se haya generado correctamente
                if not file_path.exists():
                    raise RuntimeError(f"Failed to generate {filename}")
                if file_path.stat().st_size == 0:
                    raise RuntimeError(f"Generated file {filename} is empty")
                    
                logger.info(f"Successfully generated {filename}")
                
            except Exception as e:
                logger.error(f"Error generating {filename}: {str(e)}")
                raise RuntimeError(f"Failed to generate {filename}: {str(e)}")
                
            finally:
                plt.close('all')
        
        # Verificación final
        all_files = list(output_path.glob('*.png'))
        logger.info(f"Generated files: {[f.name for f in all_files]}")
        
        if len(all_files) != len(plots_to_generate):
            missing = set(f[1] for f in plots_to_generate) - set(f.name for f in all_files)
            raise RuntimeError(f"Some files were not generated: {missing}")
        
def plot_correlation_matrix(matrix: np.ndarray,
                          title: str,
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create heatmap visualization of correlation/covariance matrix.
    
    Args:
        matrix: Correlation or covariance matrix
        title: Plot title
        ax: Optional matplotlib axes
        
    Returns:
        matplotlib axes with plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
        
    mask = np.triu(np.ones_like(matrix), k=1)
    sns.heatmap(matrix,
                mask=mask,
                cmap='RdYlBu_r',
                ax=ax,
                center=0,
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Correlation'})
                
    ax.set_title(title)
    return ax