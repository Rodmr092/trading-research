# test_error_handling.py

import pytest
import numpy as np
import logging
from src.error_analysis import ErrorAnalysisConfig, ErrorAnalyzer, PortfolioParameters
from numpy.linalg import LinAlgError

@pytest.fixture
def base_config():
    """Create a minimal test configuration."""
    return ErrorAnalysisConfig(
        n_iterations=2,
        error_magnitudes=np.array([0.05]),
        risk_tolerances=np.array([25]),
        n_jobs=1,
        random_seed=42
    )

def test_config_validation(base_config):
    """Test configuration parameter validation."""
    with pytest.raises(ValueError, match="n_iterations must be positive"):
        ErrorAnalysisConfig(n_iterations=0)
    
    with pytest.raises(ValueError, match="n_iterations must be positive"):
        ErrorAnalysisConfig(n_iterations=-1)

def test_invalid_dimensions(base_config):
    """Test handling of mismatched dimensions."""
    analyzer = ErrorAnalyzer(base_config)
    
    invalid_params = PortfolioParameters(
        expected_returns=np.array([1, 2]),
        covariance_matrix=np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
    )
    
    with pytest.raises(ValueError, match="Dimension mismatch"):
        analyzer.analyze_errors(invalid_params)

def test_invalid_covariance(base_config):
    """Test handling of invalid covariance matrix."""
    analyzer = ErrorAnalyzer(base_config)
    
    invalid_cov = np.array([
        [1.0, 2.0, 2.0],
        [2.0, 1.0, 2.0],
        [2.0, 2.0, 1.0]
    ])
    
    invalid_params = PortfolioParameters(
        expected_returns=np.array([1, 1, 1]),
        covariance_matrix=invalid_cov
    )
    
    with pytest.raises(ValueError, match="Covariance matrix is not positive definite"):
        analyzer.analyze_errors(invalid_params)

def test_invalid_values(base_config):
    """Test handling of invalid values in covariance matrix."""
    analyzer = ErrorAnalyzer(base_config)
    
    # Verificar primero los NaN
    invalid_cov = np.array([
        [1.0, np.nan, 0.5],
        [np.nan, 1.0, 0.5],
        [0.5, 0.5, 1.0]
    ])
    
    invalid_params = PortfolioParameters(
        expected_returns=np.array([1, 1, 1]),
        covariance_matrix=invalid_cov
    )
    
    # Aceptar tanto ValueError como LinAlgError
    with pytest.raises((ValueError, LinAlgError)):
        analyzer.analyze_errors(invalid_params)
    
    # Verificar infinitos
    invalid_cov_inf = np.array([
        [1.0, np.inf, 0.5],
        [np.inf, 1.0, 0.5],
        [0.5, 0.5, 1.0]
    ])
    
    invalid_params_inf = PortfolioParameters(
        expected_returns=np.array([1, 1, 1]),
        covariance_matrix=invalid_cov_inf
    )
    
    with pytest.raises((ValueError, LinAlgError)):
        analyzer.analyze_errors(invalid_params_inf)

def test_optimization_failure(base_config, caplog):
    """Test handling of optimization failures."""
    analyzer = ErrorAnalyzer(base_config)
    caplog.set_level(logging.ERROR)
    
    # Crear una matriz de covarianza que cause problemas numéricos
    problematic_cov = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    
    problematic_params = PortfolioParameters(
        expected_returns=np.array([0.1, 0.1, 0.1]),  # Retornos idénticos
        covariance_matrix=problematic_cov  # Matriz singular
    )
    
    # Probar _run_single_simulation
    result = analyzer._run_single_simulation(
        ('means', 0.1, 25, problematic_params, 42)
    )
    assert result is None, "Expected None result for problematic optimization"
    
    # Probar analyze_errors
    with pytest.raises(ValueError, match="Covariance matrix is not positive definite"):
        analyzer.analyze_errors(problematic_params)
    
    # Verificar los mensajes de error en el log
    assert any("Error in simulation" in record.message 
              for record in caplog.records), "Expected error message in logs"

def test_optimization_numerical_instability(base_config, caplog):
    """Test handling of numerically unstable optimizations."""
    analyzer = ErrorAnalyzer(base_config)
    caplog.set_level(logging.ERROR)
    
    # Crear una matriz casi singular pero técnicamente definida positiva
    base_matrix = np.ones((3, 3))
    epsilon = 1e-10
    problematic_cov = base_matrix + np.eye(3) * epsilon
    
    problematic_params = PortfolioParameters(
        expected_returns=np.array([1.0, 1.0 + epsilon, 1.0 - epsilon]),
        covariance_matrix=problematic_cov
    )
    
    # La optimización debería fallar debido a la inestabilidad numérica
    result = analyzer._run_single_simulation(
        ('means', 0.1, 25, problematic_params, 42)
    )
    
    # Verificar que el resultado es None debido a problemas numéricos
    assert result is None, "Expected None result for numerically unstable optimization"
    
    # Verificar que se registró un error
    assert any("Error in simulation" in record.message or 
              "Optimization error" in record.message 
              for record in caplog.records), "Expected error message in logs"
    
    
if __name__ == '__main__':
    pytest.main([__file__, '-v'])