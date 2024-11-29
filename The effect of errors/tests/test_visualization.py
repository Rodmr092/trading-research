"""Test suite for visualization.py module"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from typing import Optional, Dict, List, Tuple, Union


from src.visualization import PortfolioVisualizer, plot_correlation_matrix

@pytest.fixture(scope='session', autouse=True)
def setup_matplotlib():
    """Configure matplotlib for testing"""
    plt.switch_backend('Agg')  # Use non-interactive backend for testing

@pytest.fixture
def sample_results_df():
    """Create sample results DataFrame for testing."""
    index = pd.MultiIndex.from_product([
        ['means', 'variances', 'covariances'],
        [0.05, 0.10, 0.15],
        [25, 50, 75]
    ], names=['error_type', 'error_magnitude', 'risk_tolerance'])
    
    np.random.seed(42)
    
    # Crear datos con estructura correcta
    data = {
        ('cel', 'mean'): np.random.uniform(0, 0.2, len(index)),
        ('cel', 'std'): np.random.uniform(0, 0.05, len(index)),
        'mean_weight_diff': np.random.uniform(0, 0.1, len(index)),
        'optimal_risk': np.random.uniform(0.1, 0.3, len(index)),
        'optimal_return': np.random.uniform(0.05, 0.15, len(index)),
        'suboptimal_risk': np.random.uniform(0.1, 0.3, len(index)),
        'suboptimal_return': np.random.uniform(0.05, 0.15, len(index))
    }
    
    # Convertir a DataFrame con MultiIndex en columnas para 'cel'
    df = pd.DataFrame(data, index=index)
    return df

@pytest.fixture
def visualizer():
    """Create PortfolioVisualizer instance."""
    return PortfolioVisualizer(style='default')

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "test_figures"
    output_dir.mkdir()
    return output_dir

def test_visualizer_initialization():
    """Test PortfolioVisualizer initialization."""
    viz = PortfolioVisualizer()
    assert viz.figsize == (10, 6)
    assert len(viz.colors) == 3
    assert all(color in viz.colors for color in ['means', 'variances', 'covariances'])

def test_plot_cel_heatmap(visualizer, sample_results_df, temp_output_dir):
    """Test CEL heatmap generation."""
    output_path = temp_output_dir / "cel_heatmap.png"
    
    visualizer.plot_cel_heatmap(sample_results_df, str(output_path))
    assert output_path.exists()
    
    plt.close('all')

def test_plot_cel_boxplots(visualizer, sample_results_df, temp_output_dir):
    """Test CEL boxplot generation."""
    output_path = temp_output_dir / "cel_boxplots.png"
    
    visualizer.plot_cel_boxplots(sample_results_df, str(output_path))
    assert output_path.exists()
    
    plt.close('all')

def test_plot_weight_differences(visualizer, sample_results_df, temp_output_dir):
    """Test weight differences plot generation."""
    output_path = temp_output_dir / "weight_differences.png"
    
    visualizer.plot_weight_differences(sample_results_df, str(output_path))
    assert output_path.exists()
    
    plt.close('all')

def test_plot_risk_return_scatter(visualizer, sample_results_df, temp_output_dir):
    """Test risk-return scatter plot generation."""
    output_path = temp_output_dir / "risk_return.png"
    
    visualizer.plot_risk_return_scatter(sample_results_df, str(output_path))
    assert output_path.exists()
    
    plt.close('all')

def test_plot_cel_confidence_bands(visualizer, sample_results_df, temp_output_dir):
    """Test CEL confidence bands plot generation."""
    output_path = temp_output_dir / "cel_confidence.png"
    
    try:
        visualizer.plot_cel_confidence_bands(sample_results_df, save_path=str(output_path))
        assert output_path.exists()
    except Exception as e:
        pytest.fail(f"Failed to generate confidence bands plot: {str(e)}")
    
    plt.close('all')
    
def test_plot_cel_confidence_bands_data_structure(visualizer, sample_results_df, temp_output_dir):
    """Test data structure for confidence bands plot."""
    # Verificar estructura del DataFrame
    assert isinstance(sample_results_df.index, pd.MultiIndex)
    assert all(name in sample_results_df.index.names 
              for name in ['error_type', 'error_magnitude', 'risk_tolerance'])
    
    # Verificar columnas necesarias
    assert ('cel', 'mean') in sample_results_df.columns
    assert ('cel', 'std') in sample_results_df.columns
    
    # Verificar tipos de error
    error_types = sample_results_df.index.get_level_values('error_type').unique()
    assert all(et in error_types for et in ['means', 'variances', 'covariances'])
    
    # Intentar generar el plot
    output_path = temp_output_dir / "test_confidence.png"
    visualizer.plot_cel_confidence_bands(sample_results_df, save_path=str(output_path))
    
    assert output_path.exists(), "Confidence bands plot was not generated"
    assert output_path.stat().st_size > 0, "Confidence bands plot file is empty"

def test_create_analysis_dashboard(visualizer, sample_results_df, temp_output_dir):
    """Test complete dashboard generation."""
    try:
        visualizer.create_analysis_dashboard(sample_results_df, str(temp_output_dir))
    except Exception as e:
        # Listar archivos que sí se generaron
        generated_files = list(temp_output_dir.glob('*.png'))
        
        pytest.fail(
            f"Dashboard generation failed with error: {str(e)}\n"
            f"Directory contents: {[f.name for f in generated_files]}"
        )
    
    expected_files = {
        'cel_heatmap.png',
        'cel_boxplots.png',
        'weight_differences.png',
        'risk_return.png',
        'cel_confidence.png'
    }
    
    # Verificar cada archivo
    for file in expected_files:
        file_path = temp_output_dir / file
        assert file_path.exists(), f"File {file} was not generated"
        assert file_path.stat().st_size > 0, f"File {file} is empty"
    
    plt.close('all')

def test_plot_correlation_matrix():
    """Test correlation matrix plot function."""
    matrix = np.array([[1.0, 0.5, 0.3],
                      [0.5, 1.0, 0.2],
                      [0.3, 0.2, 1.0]])
    
    fig, ax = plt.subplots()
    plot_correlation_matrix(matrix, "Test Matrix", ax)
    
    assert ax.get_title() == "Test Matrix"
    assert len(ax.collections) > 0  # Verify heatmap was created
    
    plt.close('all')

def test_invalid_data_handling(visualizer, temp_output_dir):
    """Test handling of invalid input data."""
    invalid_df = pd.DataFrame()
    
    with pytest.raises(Exception):
        visualizer.plot_cel_heatmap(invalid_df, str(temp_output_dir / "test.png"))

@pytest.mark.parametrize("error_type", ['means', 'variances', 'covariances'])
def test_color_consistency(visualizer, error_type):
    """Test color consistency across plots."""
    assert error_type in visualizer.colors
    assert visualizer.colors[error_type].startswith('#')

def test_figure_setup(visualizer):
    """Test figure setup helper function."""
    title = "Test Plot"
    fig, ax = visualizer.setup_figure(title)
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == title
    
    plt.close('all')