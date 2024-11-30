import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from src.error_analysis import ErrorAnalysisConfig, ErrorAnalyzer, run_error_analysis, SimulationProgress
from src.portfolio_optimizer import PortfolioParameters, PortfolioOptimizer  # Añadido PortfolioOptimizer
import logging
from numpy.linalg import LinAlgError
from .test_utils import generate_test_covariance_matrix


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
        error_magnitudes=np.array([0.05, 0.15]),
        risk_tolerances=np.array([25, 75]),
        n_jobs=2,
        random_seed=42,
        batch_size=2,
        show_progress=False  # Deshabilitar progreso en tests
    )

# ---------------------- Config Tests ----------------------

def test_config_custom_initialization():
    """Test custom initialization of ErrorAnalysisConfig."""
    custom_config = ErrorAnalysisConfig(
        n_iterations=50,
        error_magnitudes=np.array([0.01, 0.02]),
        risk_tolerances=np.array([10, 20]),
        n_jobs=4,
        random_seed=42,
        batch_size=50,
        show_progress=True
    )
    assert custom_config.n_iterations == 50
    np.testing.assert_array_equal(custom_config.error_magnitudes, np.array([0.01, 0.02]))
    np.testing.assert_array_equal(custom_config.risk_tolerances, np.array([10, 20]))
    assert custom_config.n_jobs == 4
    assert custom_config.random_seed == 42
    assert custom_config.batch_size == 50
    assert custom_config.show_progress is True

#  ---------------------- ErrorAnalyzer Tests ----------------------

def test_error_analyzer_initialization(sample_config):
    """Test initialization of ErrorAnalyzer."""
    analyzer = ErrorAnalyzer(sample_config)
    assert analyzer.config == sample_config
    assert isinstance(analyzer.base_results, dict)
    assert len(analyzer.base_results) == 0
    assert isinstance(analyzer.rng, np.random.RandomState)

def test_error_params_generation(sample_portfolio_params):
    """Test generation of error parameters for all error types."""
    analyzer = ErrorAnalyzer()
    error_magnitude = 0.1
    batch_size = 3
    
    for error_type in ['means', 'variances', 'covariances']:
        error_params_list = analyzer._generate_error_params(
            error_type,
            sample_portfolio_params,
            error_magnitude,
            batch_size,
            seed=42
        )
        
        assert isinstance(error_params_list, list)
        assert len(error_params_list) == batch_size
        
        for error_params in error_params_list:
            assert isinstance(error_params, PortfolioParameters)
            
            # Check dimensions
            assert len(error_params.expected_returns) == len(sample_portfolio_params.expected_returns)
            assert error_params.covariance_matrix.shape == sample_portfolio_params.covariance_matrix.shape
            
            # Check matrix properties
            cov_matrix = error_params.covariance_matrix
            eigenvals = np.linalg.eigvals(cov_matrix)
            assert np.all(eigenvals > -1e-10)  # Positive definite
            assert np.allclose(cov_matrix, cov_matrix.T)  # Symmetric

def test_simulation_progress():
    """Test SimulationProgress class functionality."""
    progress = SimulationProgress(total_simulations=100, show_progress=False)
    
    # Test updates
    progress.update(10, True)
    assert progress.results_count == 10
    assert progress.error_count == 0
    
    progress.update(5, False)
    assert progress.results_count == 10
    assert progress.error_count == 5
    
    # Test statistics
    stats = progress.get_stats()
    assert stats['successful_simulations'] == 10
    assert stats['failed_simulations'] == 5
    assert 'elapsed_time' in stats
    assert 'simulations_per_second' in stats

def test_config_total_simulations():
    """Test total_simulations calculation."""
    config = ErrorAnalysisConfig(
        n_iterations=50,
        error_magnitudes=np.array([0.05, 0.15]),
        risk_tolerances=np.array([25, 75])
    )
    
    # 3 error types × 2 magnitudes × 2 tolerances × 50 iterations
    expected_total = 3 * 2 * 2 * 50
    assert config.total_simulations() == expected_total

def test_batch_simulation(sample_portfolio_params, sample_config):
    """Test batch simulation functionality."""
    analyzer = ErrorAnalyzer(sample_config)
    
    for error_type in ['means', 'variances', 'covariances']:
        results, successes, failures = analyzer._run_batch_simulation(
            (error_type, 0.1, 50, sample_portfolio_params, 42, 2)
        )
        
        assert isinstance(results, list)
        assert isinstance(successes, int)
        assert isinstance(failures, int)
        assert successes + failures == 2  # batch_size
        
        for result in results:
            assert isinstance(result, dict)
            assert 'error_type' in result
            assert 'error_magnitude' in result
            assert 'risk_tolerance' in result
            assert 'cel' in result
            assert 'max_weight_diff' in result
            assert 'mean_weight_diff' in result

def test_parallel_execution(sample_portfolio_params, sample_config):
    """Test parallel execution of simulations."""
    analyzer = ErrorAnalyzer(sample_config)
    results = analyzer.analyze_errors(sample_portfolio_params)
    
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert 'cel' in results.columns.levels[0]
    assert set(['mean', 'std', 'min', 'max']).issubset(results.columns.levels[1])

# ---------------------- Error Handling Tests ----------------------

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
    with pytest.raises(ValueError):  # Removido el match específico
        analyzer.analyze_errors(invalid_params)

# ---------------------- Integration Tests ----------------------

def test_full_analysis_integration():
    """Test full integration of error analysis functionality."""
    # Setup
    n_assets = 10
    expected_returns = np.random.normal(0.1, 0.05, n_assets)
    cov_matrix = generate_test_covariance_matrix(n_assets)
    
    config = ErrorAnalysisConfig(
        n_iterations=10,
        error_magnitudes=np.array([0.05, 0.15]),
        risk_tolerances=np.array([25, 75]),
        batch_size=5,
        show_progress=False,
        random_seed=42
    )
    
    # Run analysis
    results = run_error_analysis(expected_returns, cov_matrix, config)
    
    # Validate results
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    
    # Check structure
    expected_columns = {
        'cel': ['mean', 'std', 'min', 'max'],
        'max_weight_diff': ['mean', 'max'],
        'mean_weight_diff': ['mean'],
        'active_positions': ['mean']
    }
    
    for col, aggs in expected_columns.items():
        for agg in aggs:
            assert (col, agg) in results.columns, f"Missing column: ({col}, {agg})"
    
    # Check value ranges
    assert np.all(results[('cel', 'mean')] >= 0)
    assert np.all(results[('cel', 'mean')] < 1)
    assert np.all(results[('max_weight_diff', 'mean')] >= 0)
    assert np.all(results[('max_weight_diff', 'mean')] <= 1)

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