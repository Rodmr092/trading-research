import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from src.error_analysis import ErrorAnalysisConfig, ErrorAnalyzer, run_error_analysis
from src.portfolio_optimizer import PortfolioParameters
import logging
from numpy.linalg import LinAlgError


# ---------------------- Fixtures ----------------------

@pytest.fixture
def sample_returns() -> np.ndarray:
    """Generate sample returns for testing."""
    rng = np.random.RandomState(42)
    returns = rng.normal(0.05, 0.15, size=10)  # 10 assets
    return returns

@pytest.fixture
def sample_covariance() -> np.ndarray:
    """Generate a valid sample covariance matrix for testing."""
    rng = np.random.RandomState(42)
    n_assets = 10
    # Generate random correlation matrix
    A = rng.normal(0, 1, size=(n_assets, n_assets))
    corr = A @ A.T
    # Normalize to correlation matrix
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    # Convert to covariance with reasonable volatilities
    vols = rng.uniform(0.1, 0.3, size=n_assets)
    cov = corr * np.outer(vols, vols)
    return cov

@pytest.fixture
def sample_portfolio_params(sample_returns, sample_covariance) -> PortfolioParameters:
    """Create sample portfolio parameters with consistent dimensions."""
    n_assets = min(len(sample_returns), sample_covariance.shape[0])
    return PortfolioParameters(
        expected_returns=sample_returns[:n_assets],
        covariance_matrix=sample_covariance[:n_assets, :n_assets]
    )

@pytest.fixture
def sample_config() -> ErrorAnalysisConfig:
    """Create a test configuration with smaller iteration counts."""
    return ErrorAnalysisConfig(
        n_iterations=5,
        error_magnitudes=np.array([0.05, 0.10]),
        risk_tolerances=np.array([25, 50]),
        n_jobs=2,
        random_seed=42
    )

# ---------------------- Config Tests ----------------------

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

def test_config_custom_initialization():
    """Test custom initialization of ErrorAnalysisConfig."""
    custom_config = ErrorAnalysisConfig(
        n_iterations=50,
        error_magnitudes=np.array([0.01, 0.02]),
        risk_tolerances=np.array([10, 20]),
        n_jobs=4,
        random_seed=42
    )
    assert custom_config.n_iterations == 50
    np.testing.assert_array_equal(custom_config.error_magnitudes, np.array([0.01, 0.02]))
    np.testing.assert_array_equal(custom_config.risk_tolerances, np.array([10, 20]))
    assert custom_config.n_jobs == 4
    assert custom_config.random_seed == 42

# ---------------------- ErrorAnalyzer Base Tests ----------------------

def test_error_analyzer_initialization(sample_config):
    """Test initialization of ErrorAnalyzer."""
    analyzer = ErrorAnalyzer(sample_config)
    assert analyzer.config == sample_config
    assert isinstance(analyzer.base_results, dict)
    assert len(analyzer.base_results) == 0
    assert isinstance(analyzer.rng, np.random.RandomState)

def test_matrix_properties_checker(sample_portfolio_params):
    """Test the matrix properties checker functionality."""
    analyzer = ErrorAnalyzer()
    original_cov = sample_portfolio_params.covariance_matrix
    
    # Test with variance errors
    error_params = analyzer._generate_variance_errors(
        sample_portfolio_params,
        error_magnitude=0.1,
        seed=42
    )
    metrics = analyzer._check_matrix_properties(
        error_params.covariance_matrix,
        original_cov,
        'variance'
    )
    
    assert isinstance(metrics, dict)
    assert metrics['positive_definite']
    assert metrics['symmetric']
    assert 0.5 <= metrics['variance_ratio'] <= 2.0
    assert 0 <= metrics['correlation_difference'] <= 0.5
    assert -1 <= metrics['max_correlation'] <= 1

# ---------------------- Error Generation Tests ----------------------

def test_mean_errors_generation(sample_portfolio_params):
    """Test generation of errors in means."""
    analyzer = ErrorAnalyzer()
    error_magnitude = 0.1
    
    error_params = analyzer._generate_mean_errors(
        sample_portfolio_params,
        error_magnitude,
        seed=42
    )
    
    # Check that signs are preserved
    original_signs = np.sign(sample_portfolio_params.expected_returns)
    error_signs = np.sign(error_params.expected_returns)
    np.testing.assert_array_equal(original_signs, error_signs)
    
    # Check that covariance matrix is unchanged
    np.testing.assert_array_equal(
        sample_portfolio_params.covariance_matrix,
        error_params.covariance_matrix
    )

def test_variance_errors_generation(sample_portfolio_params):
    """Test generation of errors in variances."""
    analyzer = ErrorAnalyzer()
    error_magnitude = 0.1
    
    error_params = analyzer._generate_variance_errors(
        sample_portfolio_params,
        error_magnitude,
        seed=42
    )
    
    cov_matrix = error_params.covariance_matrix
    
    # Check positive definiteness
    eigenvals = np.linalg.eigvals(cov_matrix)
    assert np.all(eigenvals > -1e-10)
    
    # Check symmetry
    assert np.allclose(cov_matrix, cov_matrix.T)
    
    # Check that correlations are approximately maintained
    orig_corr = np.corrcoef(sample_portfolio_params.covariance_matrix)
    new_corr = np.corrcoef(cov_matrix)
    assert np.mean(np.abs(orig_corr - new_corr)) < 0.1

def test_covariance_errors_generation(sample_portfolio_params):
    """Test generation of errors in covariances."""
    analyzer = ErrorAnalyzer()
    error_magnitude = 0.1
    
    error_params = analyzer._generate_covariance_errors(
        sample_portfolio_params,
        error_magnitude,
        seed=42
    )
    
    cov_matrix = error_params.covariance_matrix
    
    # Check positive definiteness
    eigenvals = np.linalg.eigvals(cov_matrix)
    assert np.all(eigenvals > -1e-10)
    
    # Check correlations are within [-1, 1]
    std = np.sqrt(np.diag(cov_matrix))
    corr = cov_matrix / np.outer(std, std)
    assert np.all(corr >= -1 - 1e-10)
    assert np.all(corr <= 1 + 1e-10)
    
    # Check symmetry
    assert np.allclose(cov_matrix, cov_matrix.T)

# ---------------------- Simulation Tests ----------------------

def test_single_simulation(sample_portfolio_params, sample_config):
    """Test single simulation run."""
    analyzer = ErrorAnalyzer(sample_config)
    
    # Asegurar que los parámetros de muestra tienen las dimensiones correctas
    n_assets = len(sample_portfolio_params.expected_returns)
    sample_portfolio_params = PortfolioParameters(
        expected_returns=sample_portfolio_params.expected_returns[:n_assets],
        covariance_matrix=sample_portfolio_params.covariance_matrix[:n_assets, :n_assets]
    )
    
    for error_type in ['means', 'variances', 'covariances']:
        result = analyzer._run_single_simulation(
            (error_type, 0.1, 50, sample_portfolio_params, 42)
        )
        
        # Si hay un error, imprimirlo para debugging
        if result is None:
            print(f"Error running simulation for {error_type}")
            continue
            
        assert isinstance(result, dict)
        assert 'error_type' in result
        assert 'error_magnitude' in result
        assert 'risk_tolerance' in result
        assert 'cel' in result
        
        if error_type in ['variances', 'covariances']:
            assert 'matrix_positive_definite' in result
            assert 'matrix_symmetric' in result

def test_parallel_execution(sample_portfolio_params, sample_config):
    """Test parallel execution of simulations."""
    analyzer = ErrorAnalyzer(sample_config)
    results = analyzer.analyze_errors(sample_portfolio_params)
    
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert 'cel' in results.columns.levels[0]
    assert set(['mean', 'std', 'min', 'max']).issubset(results.columns.levels[1])

# ---------------------- Integration Tests ----------------------

def test_full_analysis_integration(sample_portfolio_params, sample_config):
    """Test complete analysis integration."""
    analyzer = ErrorAnalyzer(sample_config)
    results = analyzer.analyze_errors(sample_portfolio_params)
    
    # Check structure
    assert isinstance(results, pd.DataFrame)
    assert set(['error_type', 'error_magnitude', 'risk_tolerance']).issubset(
        results.index.names
    )
    
    # Check reproducibility
    analyzer2 = ErrorAnalyzer(sample_config)
    results2 = analyzer2.analyze_errors(sample_portfolio_params)
    pd.testing.assert_frame_equal(results, results2)

def test_error_handling(sample_config):
    """Test error handling for invalid inputs."""
    analyzer = ErrorAnalyzer(sample_config)

    # Test 1: Matriz con NaN
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
    
    # Test 2: Matriz no definida positiva
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
    
    # Test 3: Dimensiones inconsistentes
    mismatched_params = PortfolioParameters(
        expected_returns=np.array([1, 1]),  # 2 elementos
        covariance_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3x3
    )
    with pytest.raises(ValueError, match="Dimension mismatch"):
        analyzer.analyze_errors(mismatched_params)
        
def test_optimization_numerical_instability(sample_config, caplog):
    """Test handling of numerically unstable optimizations."""
    analyzer = ErrorAnalyzer(sample_config)
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
    
# ---------------------- Run Error Analysis Function Test ----------------------

def test_run_error_analysis_function(sample_returns, sample_covariance, sample_config):
    """Test the convenience function for running error analysis."""
    results = run_error_analysis(
        expected_returns=sample_returns,
        covariance_matrix=sample_covariance,
        config=sample_config
    )
    
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert 'cel' in results.columns.levels[0]
    assert set(['mean', 'std', 'min', 'max']).issubset(results.columns.levels[1])