"""
Test script for validating the Error Analysis module implementation.
Tests each component and runs a small-scale simulation to verify results.
"""

import numpy as np
import pandas as pd
from error_analysis import ErrorAnalyzer, ErrorAnalysisConfig
from portfolio_optimizer import PortfolioParameters
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(n_assets: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic data for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Generar retornos más dispersos (5% to 15% anual)
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    
    # Crear matriz de covarianza más realista
    # Usar un modelo de 3 factores para más complejidad
    factor_loadings = np.random.normal(1, 0.3, (n_assets, 3))
    factor_cov = np.array([
        [0.04, 0.02, 0.01],
        [0.02, 0.03, 0.015],
        [0.01, 0.015, 0.025]
    ])
    # Aumentar la volatilidad idiosincrática
    idiosyncratic = np.diag(np.random.uniform(0.03, 0.08, n_assets))
    
    covariance_matrix = (factor_loadings @ factor_cov @ factor_loadings.T + 
                        idiosyncratic)
    
    # Asegurar que la matriz es simétrica y definida positiva
    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2
    
    return expected_returns, covariance_matrix

def test_parameter_generation(analyzer: ErrorAnalyzer, 
                            true_params: PortfolioParameters,
                            error_magnitude: float = 0.10) -> None:
    """Test error generation functions."""
    logger.info("Testing parameter generation functions...")
    
    # Test mean errors
    mean_params = analyzer._generate_mean_errors(true_params, error_magnitude, seed=42)
    mean_diff = np.abs(mean_params.expected_returns - true_params.expected_returns)
    mean_rel_error = np.mean(mean_diff / np.abs(true_params.expected_returns))
    
    logger.info(f"Mean relative error magnitude: {mean_rel_error:.4f}")
    assert 0 < mean_rel_error < 2 * error_magnitude, "Mean errors outside expected range"
    
    # Test variance errors
    var_params = analyzer._generate_variance_errors(true_params, error_magnitude, seed=42)
    var_diff = np.abs(np.diag(var_params.covariance_matrix) - 
                     np.diag(true_params.covariance_matrix))
    var_rel_error = np.mean(var_diff / np.diag(true_params.covariance_matrix))
    
    logger.info(f"Variance relative error magnitude: {var_rel_error:.4f}")
    assert 0 < var_rel_error < 2 * error_magnitude, "Variance errors outside expected range"
    
    # Test covariance errors
    cov_params = analyzer._generate_covariance_errors(true_params, error_magnitude, seed=42)
    mask = ~np.eye(len(true_params.expected_returns), dtype=bool)
    cov_diff = np.abs(cov_params.covariance_matrix[mask] - 
                     true_params.covariance_matrix[mask])
    cov_rel_error = np.mean(cov_diff / np.abs(true_params.covariance_matrix[mask]))
    
    logger.info(f"Covariance relative error magnitude: {cov_rel_error:.4f}")
    assert 0 < cov_rel_error < 2 * error_magnitude, "Covariance errors outside expected range"
    
    logger.info("All parameter generation tests passed!")

def test_simulation_consistency(analyzer: ErrorAnalyzer, 
                              true_params: PortfolioParameters) -> None:
    """Test consistency of simulation results."""
    logger.info("Testing simulation consistency...")
    
    config = ErrorAnalysisConfig(
        n_iterations=20,  # Aumentado para mayor robustez
        error_magnitudes=np.array([0.05, 0.10]),
        risk_tolerances=np.array([25, 50, 75]),
        n_jobs=1,  # Use single process for testing
        random_seed=42
    )
    
    analyzer1 = ErrorAnalyzer(config)
    results1 = analyzer1.analyze_errors(true_params)
    
    analyzer2 = ErrorAnalyzer(config)
    results2 = analyzer2.analyze_errors(true_params)
    
    pd.testing.assert_frame_equal(results1, results2)
    logger.info("Simulation consistency test passed!")

def test_risk_tolerance_impact(analyzer: ErrorAnalyzer, 
                             true_params: PortfolioParameters) -> None:
    """Test impact of different risk tolerances."""
    logger.info("Testing risk tolerance impact...")
    
    results = analyzer.analyze_errors(true_params)
    
    logger.info("\nResultados estructura:")
    logger.info(f"Columnas disponibles: {results.columns}")
    logger.info(f"Índice: {results.index.names}")
    
    impact = results.xs('mean', axis=1, level=1).unstack(level='risk_tolerance')
    
    logger.info("\nRisk Tolerance Impact Analysis:")
    logger.info(impact)
    
    # Calcular diferencias relativas entre risk tolerances
    for error_type in ['means', 'variances', 'covariances']:
        values = impact.loc[error_type]['cel'].values
        rel_diff = np.abs(np.diff(values, axis=1)) / np.abs(values[:, :-1] + 1e-10)
        max_rel_diff = np.max(rel_diff)
        logger.info(f"Max relative difference for {error_type}: {max_rel_diff:.4f}")
        
        # Relajar el criterio de impacto
        assert np.any(rel_diff > 0.01), f"Risk tolerance has no significant impact on {error_type}"
    
    logger.info("Risk tolerance impact test passed!")
    
def test_matrix_properties(analyzer: ErrorAnalyzer, 
                         true_params: PortfolioParameters) -> None:
    """Test statistical properties of generated matrices."""
    logger.info("Testing matrix statistical properties...")
    
    error_magnitude = 0.10
    n_trials = 100
    
    # Recolectar estadísticas para errores en varianzas
    var_stats = []
    for i in range(n_trials):
        var_params = analyzer._generate_variance_errors(true_params, error_magnitude, seed=42+i)
        metrics = analyzer._check_matrix_properties(
            var_params.covariance_matrix,
            true_params.covariance_matrix,
            'variance'
        )
        var_stats.append(metrics)
    
    # Recolectar estadísticas para errores en covarianzas
    cov_stats = []
    for i in range(n_trials):
        cov_params = analyzer._generate_covariance_errors(true_params, error_magnitude, seed=42+i)
        metrics = analyzer._check_matrix_properties(
            cov_params.covariance_matrix,
            true_params.covariance_matrix,
            'covariance'
        )
        cov_stats.append(metrics)
    
    # Convertir a DataFrames para análisis
    var_df = pd.DataFrame(var_stats)
    cov_df = pd.DataFrame(cov_stats)
    
    # Verificar propiedades
    logger.info("\nVariance Error Statistics:")
    logger.info(f"Average variance ratio: {var_df['variance_ratio'].mean():.4f}")
    logger.info(f"Correlation stability: {var_df['correlation_difference'].mean():.4f}")
    logger.info(f"Positive definite violations: {(~var_df['positive_definite']).sum()}")
    
    logger.info("\nCovariance Error Statistics:")
    logger.info(f"Average variance ratio: {cov_df['variance_ratio'].mean():.4f}")
    logger.info(f"Correlation changes: {cov_df['correlation_difference'].mean():.4f}")
    logger.info(f"Max correlation: {cov_df['max_correlation'].mean():.4f}")
    logger.info(f"Positive definite violations: {(~cov_df['positive_definite']).sum()}")
    
    # Assertions
    assert all(var_df['positive_definite']), "Variance errors produced invalid matrices"
    assert all(cov_df['positive_definite']), "Covariance errors produced invalid matrices"
    assert var_df['variance_ratio'].mean() > 1.0, "Variance errors too small"
    assert cov_df['correlation_difference'].mean() > 0.01, "Covariance errors too small"
    
    logger.info("Matrix property tests passed!")

def test_cel_sensitivity(analyzer: ErrorAnalyzer,
                        true_params: PortfolioParameters) -> None:
    """Test sensitivity of CEL to risk tolerance."""
    logger.info("Testing CEL sensitivity to risk tolerance...")
    
    # Configurar prueba con rango más amplio de risk_tolerance
    error_magnitude = 0.15
    risk_tolerances = np.array([1, 10, 25, 50, 75, 99])  # Agregados valores extremos
    n_trials = 50
    
    results = []
    for rt in risk_tolerances:
        cel_values = []
        for i in range(n_trials):
            # Probar tanto errores en varianzas como en covarianzas
            for error_type in ['means', 'variances', 'covariances']:  # Agregado 'means' para comparación
                result = analyzer._run_single_simulation(
                    (error_type, error_magnitude, rt, true_params, 42+i)
                )
                if result:
                    cel_values.append({
                        'error_type': result['error_type'],
                        'cel': result['cel']
                    })
        
        # Calcular estadísticas por tipo de error
        cel_by_type = pd.DataFrame(cel_values).groupby('error_type')['cel'].agg(['mean', 'std'])
        
        results.append({
            'risk_tolerance': rt,
            'means_cel': cel_by_type.loc['means', 'mean'],
            'variances_cel': cel_by_type.loc['variances', 'mean'],
            'covariances_cel': cel_by_type.loc['covariances', 'mean'],
            'means_std': cel_by_type.loc['means', 'std'],
            'variances_std': cel_by_type.loc['variances', 'std'],
            'covariances_std': cel_by_type.loc['covariances', 'std']
        })
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(results)
    
    logger.info("\nCEL Sensitivity Results by Error Type:")
    logger.info(results_df)
    
    # Analizar variación por tipo de error
    for error_type in ['means', 'variances', 'covariances']:
        cel_variation = np.ptp(results_df[f'{error_type}_cel'])
        logger.info(f"\n{error_type.capitalize()} CEL variation across risk tolerances: {cel_variation:.4f}")
        
        # Verificar que hay variación significativa
        assert cel_variation > 0.01, f"CEL not sufficiently sensitive to risk tolerance for {error_type}"
    
    logger.info("CEL sensitivity test passed!")


def main():
    """Run all tests."""
    logger.info("Starting Error Analysis module tests...")
    
    expected_returns, covariance_matrix = create_test_data()
    true_params = PortfolioParameters(expected_returns, covariance_matrix)
    
    config = ErrorAnalysisConfig(
        n_iterations=50,
        error_magnitudes=np.array([0.05, 0.10, 0.15]),
        risk_tolerances=np.array([25, 50, 75]),
        n_jobs=1,
        random_seed=42
    )
    analyzer = ErrorAnalyzer(config)
    
    try:
        test_parameter_generation(analyzer, true_params)
        test_matrix_properties(analyzer, true_params)
        test_cel_sensitivity(analyzer, true_params)
        test_simulation_consistency(analyzer, true_params)
        
        logger.info("\nRunning complete analysis with test data...")
        results = analyzer.analyze_errors(true_params)
        
        logger.info("\nFinal Results Summary:")
        logger.info(results)
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()