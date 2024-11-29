"""
Temporary test file for ErrorAnalysisConfig initialization and error handling.
"""

import pytest
import numpy as np
import pandas as pd
from src.error_analysis import ErrorAnalysisConfig, ErrorAnalyzer
from src.portfolio_optimizer import PortfolioParameters

def test_config_default_initialization():
    """Test default initialization of ErrorAnalysisConfig."""
    config = ErrorAnalysisConfig()
    
    # Validar valores por defecto
    assert config.n_iterations == 100
    np.testing.assert_array_equal(
        config.error_magnitudes,
        np.array([0.05, 0.10, 0.15, 0.20])
    )
    np.testing.assert_array_equal(
        config.risk_tolerances,
        np.array([25, 50, 75])
    )
    assert config.n_jobs > 0  # Debe ser positivo después del post_init
    assert config.random_seed is None
    
    # Probar que n_jobs se convierte correctamente
    config_neg = ErrorAnalysisConfig(n_jobs=-1)
    assert config_neg.n_jobs > 0
    
    # Probar que valores válidos se mantienen
    config_pos = ErrorAnalysisConfig(n_jobs=2)
    assert config_pos.n_jobs == 2

def test_error_matrix_validation():
    """Test error handling for invalid matrices."""
    config = ErrorAnalysisConfig(
        n_iterations=2,  # Mínimo para pruebas
        error_magnitudes=np.array([0.05]),
        risk_tolerances=np.array([25]),
        n_jobs=1  # Un solo proceso para debugging
    )
    analyzer = ErrorAnalyzer(config)
    
    # Caso 1: Matriz con NaN
    nan_cov = np.array([
        [1.0, np.nan, 0.5],
        [np.nan, 1.0, 0.5],
        [0.5, 0.5, 1.0]
    ])
    invalid_params = PortfolioParameters(
        expected_returns=np.array([1, 1, 1]),
        covariance_matrix=nan_cov
    )
    with pytest.raises(ValueError, match="Invalid values in covariance matrix"):
        analyzer.analyze_errors(invalid_params)
    
    # Caso 2: Matriz no definida positiva
    non_pd_cov = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    invalid_params = PortfolioParameters(
        expected_returns=np.array([1, 1, 1]),
        covariance_matrix=non_pd_cov
    )
    with pytest.raises(ValueError, match="Covariance matrix is not positive definite"):
        analyzer.analyze_errors(invalid_params)
    
    # Caso 3: Dimensiones inconsistentes
    mismatched_params = PortfolioParameters(
        expected_returns=np.array([1, 1]),  # 2 elementos
        covariance_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3x3
    )
    with pytest.raises(ValueError, match="Dimension mismatch"):
        analyzer.analyze_errors(mismatched_params)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])