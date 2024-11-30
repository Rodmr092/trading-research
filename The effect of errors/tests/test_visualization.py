"""Test suite for visualization.py module"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import logging

from src.visualization import PortfolioVisualizer, plot_correlation_matrix

@pytest.fixture(scope='session', autouse=True)
def setup_matplotlib():
    """Configure matplotlib for testing"""
    plt.switch_backend('Agg')  # Use non-interactive backend for testing

@pytest.fixture
def sample_results_df():
    """Create sample results DataFrame with real-world structure."""
    # Crear índice multinivel
    index = pd.MultiIndex.from_product([
        ['means', 'variances', 'covariances'],  # error_type
        [0.05, 0.20],                          # error_magnitude
        [25, 75]                               # risk_tolerance
    ], names=['error_type', 'error_magnitude', 'risk_tolerance'])
    
    # Crear columnas multinivel
    columns = pd.MultiIndex.from_tuples([
        ('cel', 'mean'),
        ('cel', 'std'),
        ('cel', 'min'),
        ('cel', 'max'),
        ('max_weight_diff', 'mean'),
        ('max_weight_diff', 'max'),
        ('mean_weight_diff', 'mean'),
        ('active_positions', 'mean')
    ])
    
    # Generar datos de ejemplo con valores realistas
    np.random.seed(42)
    n_samples = len(index)
    
    data = np.random.uniform(low=[
        0.001,  # cel mean
        0.0005, # cel std
        0.0,    # cel min
        0.005,  # cel max
        0.01,   # max_weight_diff mean
        0.02,   # max_weight_diff max
        0.005,  # mean_weight_diff mean
        3       # active_positions mean
    ], high=[
        0.02,   # cel mean
        0.005,  # cel std
        0.01,   # cel min
        0.05,   # cel max
        0.2,    # max_weight_diff mean
        0.3,    # max_weight_diff max
        0.1,    # mean_weight_diff mean
        6       # active_positions mean
    ], size=(n_samples, len(columns)))
    
    # Crear DataFrame con estructura correcta
    df = pd.DataFrame(data, index=index, columns=columns)
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
    
    try:
        visualizer.plot_cel_heatmap(sample_results_df, str(output_path))
        assert output_path.exists(), "Heatmap file not generated"
        assert output_path.stat().st_size > 0, "Heatmap file is empty"
        
        # Verificar que la imagen sea válida
        with Image.open(output_path) as img:
            assert img.size[0] > 0 and img.size[1] > 0
            
    except Exception as e:
        pytest.fail(f"Failed to generate CEL heatmap: {str(e)}")
    
    plt.close('all')

def test_plot_cel_boxplots(visualizer, sample_results_df, temp_output_dir):
    """Test CEL boxplot generation."""
    output_path = temp_output_dir / "cel_boxplots.png"
    
    try:
        visualizer.plot_cel_boxplots(sample_results_df, str(output_path))
        assert output_path.exists(), "Boxplot file not generated"
        assert output_path.stat().st_size > 0, "Boxplot file is empty"
        
        with Image.open(output_path) as img:
            assert img.size[0] > 0 and img.size[1] > 0
            
    except Exception as e:
        pytest.fail(f"Failed to generate CEL boxplots: {str(e)}")
    
    plt.close('all')

def test_plot_weight_differences(visualizer, sample_results_df, temp_output_dir):
    """Test weight differences plot generation."""
    output_path = temp_output_dir / "weight_differences.png"
    
    try:
        visualizer.plot_weight_differences(sample_results_df, str(output_path))
        assert output_path.exists(), "Weight differences plot not generated"
        assert output_path.stat().st_size > 0, "Weight differences plot is empty"
        
        with Image.open(output_path) as img:
            assert img.size[0] > 0 and img.size[1] > 0
            
    except Exception as e:
        pytest.fail(f"Failed to generate weight differences plot: {str(e)}")
    
    plt.close('all')

def test_plot_cel_confidence_bands(visualizer, sample_results_df, temp_output_dir):
    """Test CEL confidence bands plot generation."""
    output_path = temp_output_dir / "cel_confidence.png"
    
    try:
        visualizer.plot_cel_confidence_bands(sample_results_df, str(output_path))
        assert output_path.exists(), "Confidence bands plot not generated"
        assert output_path.stat().st_size > 0, "Confidence bands plot is empty"
        
        with Image.open(output_path) as img:
            assert img.size[0] > 0 and img.size[1] > 0
            
    except Exception as e:
        pytest.fail(f"Failed to generate confidence bands plot: {str(e)}")
    
    plt.close('all')

def test_create_analysis_dashboard(visualizer, sample_results_df, temp_output_dir):
    """Test complete dashboard generation."""
    try:
        visualizer.create_analysis_dashboard(sample_results_df, str(temp_output_dir))
        
        expected_files = {
            'cel_heatmap.png',
            'cel_boxplots.png',
            'weight_differences.png',
            'cel_confidence.png'
        }
        
        # Verificar cada archivo
        for file in expected_files:
            file_path = temp_output_dir / file
            assert file_path.exists(), f"File {file} was not generated"
            assert file_path.stat().st_size > 0, f"File {file} is empty"
            
            # Verificar que cada archivo sea una imagen válida
            with Image.open(file_path) as img:
                assert img.size[0] > 0 and img.size[1] > 0, f"Invalid image dimensions for {file}"
                
    except Exception as e:
        # Listar archivos que sí se generaron
        generated_files = list(temp_output_dir.glob('*.png'))
        pytest.fail(
            f"Dashboard generation failed with error: {str(e)}\n"
            f"Directory contents: {[f.name for f in generated_files]}"
        )
    
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

def test_color_consistency(visualizer):
    """Test color consistency across plots."""
    for error_type in ['means', 'variances', 'covariances']:
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

def test_data_structure_validation(sample_results_df):
    """Test that sample data has the correct structure."""
    # Verificar estructura del índice
    assert isinstance(sample_results_df.index, pd.MultiIndex)
    assert list(sample_results_df.index.names) == ['error_type', 'error_magnitude', 'risk_tolerance']
    
    # Verificar columnas requeridas
    required_columns = [
        ('cel', 'mean'),
        ('cel', 'std'),
        ('mean_weight_diff', 'mean'),
        ('active_positions', 'mean')
    ]
    
    for col in required_columns:
        assert col in sample_results_df.columns, f"Missing required column: {col}"
    
    # Verificar tipos de error
    error_types = sample_results_df.index.get_level_values('error_type').unique()
    assert all(et in error_types for et in ['means', 'variances', 'covariances'])