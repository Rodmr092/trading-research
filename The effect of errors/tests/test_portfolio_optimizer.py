import pytest
import numpy as np
from scipy.optimize import minimize  
from src.portfolio_optimizer import PortfolioParameters, PortfolioOptimizer, create_base_portfolio

# ================== Fixtures ==================
@pytest.fixture
def small_portfolio():
    """Create a small test portfolio with known values"""
    expected_returns = np.array([0.10, 0.15])
    covariance_matrix = np.array([
        [0.05, 0.02],
        [0.02, 0.08]
    ])
    return {
        'returns': expected_returns,
        'cov_matrix': covariance_matrix,
        'risk_tolerance': 2.0,
        'max_weight': 0.8
    }

@pytest.fixture
def realistic_portfolio():
    """Create a realistic test portfolio with more assets"""
    n_assets = 10
    np.random.seed(42)  # For reproducibility
    
    expected_returns = np.random.normal(0.10, 0.05, n_assets)
    
    # Create a valid covariance matrix
    A = np.random.normal(0, 1, (n_assets, n_assets))
    covariance_matrix = A.T @ A  # Ensures positive definite
    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2  # Ensures symmetry
    
    return {
        'returns': expected_returns,
        'cov_matrix': covariance_matrix,
        'risk_tolerance': 2.5,
        'max_weight': 0.4
    }

@pytest.fixture
def invalid_cov_matrix():
    """Create an invalid (non-symmetric) covariance matrix"""
    return np.array([
        [1.0, 0.2],
        [0.3, 1.0]  # Note: 0.2 != 0.3, making it non-symmetric
    ])

# ================== PortfolioParameters Tests ==================
def test_portfolio_parameters_validation_dimensions():
    """Test validation of mismatched dimensions"""
    with pytest.raises(ValueError, match="Dimension mismatch"):
        params = PortfolioParameters(
            expected_returns=np.array([0.1, 0.2]),
            covariance_matrix=np.array([[1.0]])
        )
        params.validate()

def test_portfolio_parameters_validation_symmetry():
    """Test validation of non-symmetric covariance matrix"""
    with pytest.raises(ValueError, match="not symmetric"):
        params = PortfolioParameters(
            expected_returns=np.array([0.1, 0.2]),
            covariance_matrix=np.array([
                [1.0, 0.2],
                [0.3, 1.0]
            ])
        )
        params.validate()

def test_portfolio_parameters_validation_positive_definite():
    """Test validation of non-positive definite matrix"""
    with pytest.raises(ValueError, match="not positive definite"):
        params = PortfolioParameters(
            expected_returns=np.array([0.1, 0.2]),
            covariance_matrix=np.array([
                [1.0, 2.0],
                [2.0, 1.0]
            ])
        )
        params.validate()

def test_portfolio_parameters_successful_creation(small_portfolio):
    """Test successful creation of PortfolioParameters"""
    params = PortfolioParameters(
        expected_returns=small_portfolio['returns'],
        covariance_matrix=small_portfolio['cov_matrix']
    )
    params.validate()  # Should not raise any exception

# ================== PortfolioOptimizer Tests ==================
def test_portfolio_optimizer_invalid_risk_tolerance():
    """Test initialization with invalid risk tolerance"""
    with pytest.raises(ValueError, match="Risk tolerance must be positive"):
        PortfolioOptimizer(risk_tolerance=0)

def test_portfolio_optimizer_invalid_max_weight():
    """Test initialization with invalid max_weight"""
    with pytest.raises(ValueError, match="max_weight must be between 0 and 1"):
        PortfolioOptimizer(risk_tolerance=1, max_weight=1.5)

def test_portfolio_optimizer_successful_initialization(small_portfolio):
    """Test successful initialization of PortfolioOptimizer"""
    optimizer = PortfolioOptimizer(
        risk_tolerance=small_portfolio['risk_tolerance'],
        max_weight=small_portfolio['max_weight']
    )
    assert optimizer.risk_tolerance == small_portfolio['risk_tolerance']
    assert optimizer.max_weight == small_portfolio['max_weight']

def test_calculate_utility(small_portfolio):
    """Test utility calculation with known values"""
    optimizer = PortfolioOptimizer(risk_tolerance=small_portfolio['risk_tolerance'])
    params = PortfolioParameters(
        expected_returns=small_portfolio['returns'],
        covariance_matrix=small_portfolio['cov_matrix']
    )
    
    weights = np.array([0.5, 0.5])
    utility = optimizer._calculate_utility(weights, params)
    
    # Calculate expected utility manually
    expected_return = np.dot(weights, small_portfolio['returns'])
    variance = weights @ small_portfolio['cov_matrix'] @ weights
    expected_utility = expected_return - (1 / (2 * small_portfolio['risk_tolerance'])) * variance
    
    np.testing.assert_almost_equal(utility, expected_utility)

def test_calculate_utility_risk_tolerance_override(small_portfolio):
    """Test utility calculation with risk tolerance override"""
    base_rt = 2.0
    override_rt = 3.0
    optimizer = PortfolioOptimizer(risk_tolerance=base_rt)
    params = PortfolioParameters(
        expected_returns=small_portfolio['returns'],
        covariance_matrix=small_portfolio['cov_matrix']
    )
    
    weights = np.array([0.5, 0.5])
    utility_override = optimizer._calculate_utility(weights, params, risk_tolerance=override_rt)
    utility_base = optimizer._calculate_utility(weights, params)
    
    assert utility_override != utility_base

def test_optimize_weights_sum(realistic_portfolio):
    """Test that optimized weights sum to 1"""
    optimizer = PortfolioOptimizer(
        risk_tolerance=realistic_portfolio['risk_tolerance'],
        max_weight=realistic_portfolio['max_weight']
    )
    params = PortfolioParameters(
        expected_returns=realistic_portfolio['returns'],
        covariance_matrix=realistic_portfolio['cov_matrix']
    )
    
    weights = optimizer.optimize(params)
    np.testing.assert_almost_equal(np.sum(weights), 1.0)

def test_optimize_max_weight_constraint(realistic_portfolio):
    """Test that no weight exceeds max_weight"""
    optimizer = PortfolioOptimizer(
        risk_tolerance=realistic_portfolio['risk_tolerance'],
        max_weight=realistic_portfolio['max_weight']
    )
    params = PortfolioParameters(
        expected_returns=realistic_portfolio['returns'],
        covariance_matrix=realistic_portfolio['cov_matrix']
    )
    
    weights = optimizer.optimize(params)
    assert np.all(weights <= realistic_portfolio['max_weight'])

def test_optimize_no_short_selling(realistic_portfolio):
    """Test that no weight is negative"""
    optimizer = PortfolioOptimizer(
        risk_tolerance=realistic_portfolio['risk_tolerance'],
        max_weight=realistic_portfolio['max_weight']
    )
    params = PortfolioParameters(
        expected_returns=realistic_portfolio['returns'],
        covariance_matrix=realistic_portfolio['cov_matrix']
    )
    
    weights = optimizer.optimize(params)
    assert np.all(weights >= 0)

def test_optimize_with_initial_weights(realistic_portfolio):
    """Test optimization with provided initial weights"""
    optimizer = PortfolioOptimizer(
        risk_tolerance=realistic_portfolio['risk_tolerance'],
        max_weight=realistic_portfolio['max_weight']
    )
    params = PortfolioParameters(
        expected_returns=realistic_portfolio['returns'],
        covariance_matrix=realistic_portfolio['cov_matrix']
    )
    
    initial_weights = np.ones(len(realistic_portfolio['returns'])) / len(realistic_portfolio['returns'])
    weights = optimizer.optimize(params, initial_weights=initial_weights)
    
    assert len(weights) == len(initial_weights)
    np.testing.assert_almost_equal(np.sum(weights), 1.0)

def test_optimize_small_weights_cleanup(realistic_portfolio):
    """Test that very small weights are set to zero"""
    optimizer = PortfolioOptimizer(
        risk_tolerance=realistic_portfolio['risk_tolerance'],
        max_weight=realistic_portfolio['max_weight']
    )
    params = PortfolioParameters(
        expected_returns=realistic_portfolio['returns'],
        covariance_matrix=realistic_portfolio['cov_matrix']
    )
    
    weights = optimizer.optimize(params)
    assert np.all((weights >= 1e-4) | (weights == 0))
    np.testing.assert_almost_equal(np.sum(weights), 1.0)

def test_calculate_cash_equivalent(small_portfolio):
    """Test cash equivalent calculation with known values"""
    optimizer = PortfolioOptimizer(risk_tolerance=small_portfolio['risk_tolerance'])
    params = PortfolioParameters(
        expected_returns=small_portfolio['returns'],
        covariance_matrix=small_portfolio['cov_matrix']
    )
    
    weights = np.array([0.5, 0.5])
    ce = optimizer.calculate_cash_equivalent(weights, params)
    
    # Calculate expected CE manually
    expected_return = np.dot(weights, small_portfolio['returns'])
    variance = weights @ small_portfolio['cov_matrix'] @ weights
    expected_ce = expected_return - (1 / (2 * small_portfolio['risk_tolerance'])) * variance
    
    np.testing.assert_almost_equal(ce, expected_ce)

def test_calculate_cash_equivalent_risk_tolerance_override(small_portfolio):
    """Test cash equivalent calculation with risk tolerance override"""
    base_rt = 2.0
    override_rt = 3.0
    optimizer = PortfolioOptimizer(risk_tolerance=base_rt)
    params = PortfolioParameters(
        expected_returns=small_portfolio['returns'],
        covariance_matrix=small_portfolio['cov_matrix']
    )
    
    weights = np.array([0.5, 0.5])
    ce_override = optimizer.calculate_cash_equivalent(weights, params, risk_tolerance=override_rt)
    ce_base = optimizer.calculate_cash_equivalent(weights, params)
    
    assert ce_override != ce_base
    
    # Verificar que los resultados son consistentes con el cálculo manual
    expected_return = np.dot(weights, small_portfolio['returns'])
    variance = weights @ small_portfolio['cov_matrix'] @ weights
    
    expected_ce_override = expected_return - (1 / (2 * override_rt)) * variance
    expected_ce_base = expected_return - (1 / (2 * base_rt)) * variance
    
    np.testing.assert_almost_equal(ce_override, expected_ce_override)
    np.testing.assert_almost_equal(ce_base, expected_ce_base)

def test_calculate_cel_manual_verification(small_portfolio):
    """Test CEL calculation against manual computation"""
    optimizer = PortfolioOptimizer(
        risk_tolerance=small_portfolio['risk_tolerance'],
        max_weight=small_portfolio['max_weight']  # Asegurarnos de usar el max_weight del portfolio
    )
    params = PortfolioParameters(
        expected_returns=small_portfolio['returns'],
        covariance_matrix=small_portfolio['cov_matrix']
    )
    
    # Verificar los datos de entrada
    print("\nInput Parameters:")
    print(f"Expected returns: {small_portfolio['returns']}")
    print(f"Covariance matrix:\n{small_portfolio['cov_matrix']}")
    print(f"Risk tolerance: {small_portfolio['risk_tolerance']}")
    print(f"Max weight: {optimizer.max_weight}")
    
    # Probar pesos sistemáticamente
    n_points = 1000
    best_utility = float('-inf')
    best_weights = None
    
    # Para debugging, probar algunos puntos específicos primero
    test_points = [
        [0.5, 0.5],
        [0.6, 0.4],
        [0.4, 0.6],
        [0.7, 0.3],
        [0.3, 0.7]
    ]
    
    print("\nTesting specific points:")
    for weights in test_points:
        weights = np.array(weights)
        if np.all(weights >= 0) and np.all(weights <= optimizer.max_weight) and np.isclose(np.sum(weights), 1.0):
            utility = optimizer._calculate_utility(weights, params)
            print(f"Weights: {weights}, Utility: {utility}")
            if utility > best_utility:
                best_utility = utility
                best_weights = weights.copy()
    
    print("\nStarting grid search:")
    # Búsqueda en cuadrícula con pesos válidos
    for i in range(n_points + 1):
        w1 = i / n_points
        w2 = 1 - w1
        
        if (0 <= w1 <= optimizer.max_weight and 
            0 <= w2 <= optimizer.max_weight and 
            np.isclose(w1 + w2, 1.0)):
            
            weights = np.array([w1, w2])
            utility = optimizer._calculate_utility(weights, params)
            
            if i % 100 == 0:  # Print every 100th point for monitoring
                print(f"Trying weights: {weights}, Utility: {utility}")
            
            if utility > best_utility:
                best_utility = utility
                best_weights = weights.copy()
                print(f"New best found - Weights: {weights}, Utility: {utility}")
    
    print("\nGrid search completed")
    print(f"Best weights found: {best_weights}")
    print(f"Best utility found: {best_utility}")
    
    assert best_weights is not None, "No valid weights found"
    weights_optimal = best_weights
    weights_suboptimal = np.array([0.3, 0.7])  # Estos pesos deben ser válidos para max_weight=0.8
    
    # Verificar que los pesos son válidos
    np.testing.assert_almost_equal(np.sum(weights_optimal), 1.0)
    assert np.all(weights_optimal <= optimizer.max_weight), "Weights exceed max_weight"
    assert np.all(weights_optimal >= 0), "Found negative weights"
    
    # Verificar que los pesos óptimos son realmente mejores
    utility_optimal = optimizer._calculate_utility(weights_optimal, params)
    utility_suboptimal = optimizer._calculate_utility(weights_suboptimal, params)
    
    print(f"\nVerification Results:")
    print(f"Optimal weights: {weights_optimal}")
    print(f"Optimal utility: {utility_optimal}")
    print(f"Suboptimal weights: {weights_suboptimal}")
    print(f"Suboptimal utility: {utility_suboptimal}")
    
    assert utility_optimal > utility_suboptimal, (
        f"Failed to find optimal weights. "
        f"Optimal utility ({utility_optimal:.6f}) <= "
        f"Suboptimal utility ({utility_suboptimal:.6f})"
    )
    
    # Calcular CEs y CEL
    ce_optimal = optimizer.calculate_cash_equivalent(weights_optimal, params)
    ce_suboptimal = optimizer.calculate_cash_equivalent(weights_suboptimal, params)
    cel = optimizer.calculate_cel(weights_optimal, weights_suboptimal, params)
    
    print(f"\nFinal Results:")
    print(f"CE optimal: {ce_optimal}")
    print(f"CE suboptimal: {ce_suboptimal}")
    print(f"CEL: {cel}")
    
    assert cel >= 0, "CEL should be non-negative"
    assert isinstance(cel, float), "CEL should be a float"

def test_calculate_cel_basic():
    """Test basic CEL calculation with simple case"""
    # Usar risk_tolerance más alto para mejor convergencia
    optimizer = PortfolioOptimizer(risk_tolerance=5.0, max_weight=1.0)
    
    # Caso simple con retornos y riesgos claramente diferenciados
    returns = np.array([0.05, 0.15])  # Segundo activo tiene mayor retorno
    cov_matrix = np.array([
        [0.10, 0.00],  # Sin correlación para simplificar
        [0.00, 0.10]
    ])
    
    params = PortfolioParameters(returns, cov_matrix)
    
    # Crear pesos subóptimos manualmente 
    weights_suboptimal = np.array([0.8, 0.2])  # Concentrado en el activo equivocado
    weights_optimal = np.array([0.2, 0.8])      # Concentrado en el mejor activo
    
    # Verificar que los pesos óptimos dan mayor utilidad
    utility_optimal = optimizer._calculate_utility(weights_optimal, params)
    utility_suboptimal = optimizer._calculate_utility(weights_suboptimal, params)
    assert utility_optimal > utility_suboptimal, \
        "Optimal weights should give higher utility"
    
    # Calcular CEL
    cel = optimizer.calculate_cel(weights_optimal, weights_suboptimal, params)
    
    # Verificaciones básicas del CEL
    assert cel >= 0, "CEL should be non-negative"
    assert cel <= 1, "CEL should not exceed 1"
    assert isinstance(cel, float), "CEL should be a float"

def test_calculate_cel_numerical_tolerance():
    """Test CEL calculation with numerical tolerance"""
    optimizer = PortfolioOptimizer(risk_tolerance=5.0)
    returns = np.array([0.10, 0.101])  # Muy similar retorno
    cov_matrix = np.array([
        [0.10, 0.00],
        [0.00, 0.10]
    ])
    
    params = PortfolioParameters(returns, cov_matrix)
    
    weights1 = np.array([0.5, 0.5])
    weights2 = np.array([0.49, 0.51])
    
    # Deberían tener utilidades muy similares
    cel = optimizer.calculate_cel(weights1, weights2, params)
    
    # El CEL debería ser muy pequeño pero no negativo
    assert cel >= 0, "CEL should be non-negative"
    assert cel < 0.01, "CEL should be very small for similar portfolios"

# Pruebas de casos extremos
def test_extreme_risk_tolerance():
    """Test optimization with extreme risk tolerance values"""
    # Crear un portfolio simple para las pruebas
    returns = np.array([0.10, 0.15])
    cov_matrix = np.array([
        [0.05, 0.02],
        [0.02, 0.08]
    ])
    params = PortfolioParameters(returns, cov_matrix)
    
    # Probar con risk_tolerance muy alto (cercano a infinito)
    high_rt_optimizer = PortfolioOptimizer(risk_tolerance=1e6, max_weight=1.0)
    high_rt_weights = high_rt_optimizer.optimize(params)
    
    # Con risk_tolerance muy alto, debería concentrarse en el activo de mayor retorno
    assert np.argmax(high_rt_weights) == np.argmax(returns)
    
    # Probar con risk_tolerance muy bajo (cercano a cero)
    low_rt_optimizer = PortfolioOptimizer(risk_tolerance=1e-6, max_weight=1.0)
    low_rt_weights = low_rt_optimizer.optimize(params)
    
    # Con risk_tolerance muy bajo, debería buscar la mínima varianza posible
    variance_low = low_rt_weights @ cov_matrix @ low_rt_weights
    
    # Verificar que cualquier otra combinación de pesos válida tiene mayor varianza
    test_weights = np.array([0.8, 0.2])
    variance_test = test_weights @ cov_matrix @ test_weights
    assert variance_low <= variance_test

def test_extreme_max_weight():
    """Test optimization with extreme max_weight values"""
    returns = np.array([0.10, 0.15, 0.12])
    cov_matrix = np.array([
        [0.05, 0.02, 0.01],
        [0.02, 0.08, 0.03],
        [0.01, 0.03, 0.06]
    ])
    params = PortfolioParameters(returns, cov_matrix)
    
    # Probar con max_weight muy pequeño (forzar diversificación)
    small_max_optimizer = PortfolioOptimizer(risk_tolerance=2.0, max_weight=0.4)
    small_max_weights = small_max_optimizer.optimize(params)
    assert np.all(small_max_weights <= 0.4)
    
    # Probar con max_weight = 1.0 (permitir concentración total)
    large_max_optimizer = PortfolioOptimizer(risk_tolerance=2.0, max_weight=1.0)
    large_max_weights = large_max_optimizer.optimize(params)
    
    # Verificar que al menos un peso es mayor que 0.4
    assert np.any(large_max_weights > 0.4)
    
    # Verificar que los pesos siguen sumando 1
    np.testing.assert_almost_equal(np.sum(large_max_weights), 1.0)

def test_calculate_cel_identical_portfolios(small_portfolio):
    """Test CEL calculation with identical portfolios"""
    optimizer = PortfolioOptimizer(risk_tolerance=small_portfolio['risk_tolerance'])
    params = PortfolioParameters(
        expected_returns=small_portfolio['returns'],
        covariance_matrix=small_portfolio['cov_matrix']
    )
    
    weights = np.array([0.5, 0.5])
    cel = optimizer.calculate_cel(weights, weights, params)
    
    np.testing.assert_almost_equal(cel, 0.0)

def test_calculate_cel_zero_ce(small_portfolio):
    """Test CEL calculation with CE close to zero"""
    optimizer = PortfolioOptimizer(risk_tolerance=small_portfolio['risk_tolerance'])
    params = PortfolioParameters(
        expected_returns=np.zeros_like(small_portfolio['returns']),
        covariance_matrix=small_portfolio['cov_matrix'] * 1e-10  # Hacer la varianza muy pequeña también
    )
    
    weights1 = np.array([0.5, 0.5])
    weights2 = np.array([0.3, 0.7])
    
    with pytest.raises(ValueError, match="CE₀ too close to zero"):
        optimizer.calculate_cel(weights1, weights2, params)
        
def test_calculate_cel_basic():
    """Test basic CEL calculation with simple case"""
    # Usar risk_tolerance más alto para mejor convergencia
    optimizer = PortfolioOptimizer(risk_tolerance=5.0, max_weight=1.0)
    
    # Caso simple con retornos y riesgos claramente diferenciados
    returns = np.array([0.05, 0.15])  # Segundo activo tiene mayor retorno
    cov_matrix = np.array([
        [0.10, 0.00],  # Sin correlación para simplificar
        [0.00, 0.10]
    ])
    
    params = PortfolioParameters(returns, cov_matrix)
    
    # Crear pesos subóptimos manualmente
    weights_suboptimal = np.array([0.8, 0.2])  # Concentrado en el activo equivocado
    
    # Optimizar para encontrar los pesos óptimos
    try:
        weights_optimal = optimizer.optimize(params)
    except Exception as e:
        pytest.fail(f"Optimization failed: {str(e)}")
    
    # Verificar que los pesos óptimos favorecen al segundo activo
    assert weights_optimal[1] > weights_optimal[0], \
        "Optimal weights should favor the second asset"
    
    # Calcular CEL
    cel = optimizer.calculate_cel(weights_optimal, weights_suboptimal, params)
    
    # Verificaciones básicas del CEL
    assert cel > 0, "CEL should be positive"
    assert cel <= 1, "CEL should not exceed 1"
    
    # Verificar que la utilidad óptima es mayor
    utility_optimal = optimizer._calculate_utility(weights_optimal, params)
    utility_suboptimal = optimizer._calculate_utility(weights_suboptimal, params)
    assert utility_optimal > utility_suboptimal, \
        "Optimal weights should give higher utility"

# ================== Integration Tests ==================
def test_create_base_portfolio(realistic_portfolio):
    """Test the complete base portfolio creation"""
    weights, optimizer = create_base_portfolio(
        expected_returns=realistic_portfolio['returns'],
        covariance_matrix=realistic_portfolio['cov_matrix'],
        risk_tolerance=realistic_portfolio['risk_tolerance'],
        max_weight=realistic_portfolio['max_weight']
    )
    
    # Verify outputs
    assert isinstance(optimizer, PortfolioOptimizer)
    assert len(weights) == len(realistic_portfolio['returns'])
    np.testing.assert_almost_equal(np.sum(weights), 1.0)
    assert np.all(weights >= 0)
    assert np.all(weights <= realistic_portfolio['max_weight'])

def test_end_to_end_optimization(realistic_portfolio):
    """Test the complete optimization pipeline"""
    # Create base portfolio
    weights, optimizer = create_base_portfolio(
        expected_returns=realistic_portfolio['returns'],
        covariance_matrix=realistic_portfolio['cov_matrix'],
        risk_tolerance=realistic_portfolio['risk_tolerance'],
        max_weight=realistic_portfolio['max_weight']
    )
    
    # Create suboptimal portfolio
    suboptimal_weights = np.ones_like(weights) / len(weights)
    
    # Calculate metrics
    params = PortfolioParameters(
        expected_returns=realistic_portfolio['returns'],
        covariance_matrix=realistic_portfolio['cov_matrix']
    )
    
    ce_optimal = optimizer.calculate_cash_equivalent(weights, params)
    ce_suboptimal = optimizer.calculate_cash_equivalent(suboptimal_weights, params)
    cel = optimizer.calculate_cel(weights, suboptimal_weights, params)
    
    # Verify results
    assert ce_optimal > ce_suboptimal  # Optimal portfolio should have higher CE
    assert cel > 0  # CEL should be positive
    assert isinstance(cel, float)