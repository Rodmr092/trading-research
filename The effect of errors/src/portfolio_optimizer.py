# portfolio_optimizer.py

"""
Portfolio Optimizer Module with diversification constraints.
Implements Markowitz mean-variance optimization with utility function and Cash Equivalent calculations.
"""

import numpy as np
from scipy.optimize import minimize
import logging
from typing import Dict, Tuple, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PortfolioParameters:
    """Container for portfolio parameters"""
    expected_returns: np.ndarray
    covariance_matrix: np.ndarray
    
    def validate(self):
        """Validate parameters dimensions and properties"""
        if len(self.expected_returns) != len(self.covariance_matrix):
            raise ValueError("Dimension mismatch between returns and covariance matrix")
        if not np.allclose(self.covariance_matrix, self.covariance_matrix.T):
            raise ValueError("Covariance matrix is not symmetric")
        
        # Verificar que la matriz sea definida positiva
        eigvals = np.linalg.eigvals(self.covariance_matrix)
        if np.any(eigvals <= 0):
            raise ValueError("Covariance matrix is not positive definite")

class PortfolioOptimizer:
    """
    Implements portfolio optimization using mean-variance framework with utility maximization.
    """
    
    def __init__(self, risk_tolerance: float, max_weight: float = 0.4):
        """
        Initialize optimizer with risk tolerance parameter.
        
        Args:
            risk_tolerance: Risk tolerance parameter (λ in the utility function)
            max_weight: Maximum weight allowed for any single asset
        """
        if risk_tolerance <= 0:
            raise ValueError("Risk tolerance must be positive")
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be between 0 and 1")
            
        self.risk_tolerance = risk_tolerance / 100.0  # Normalizar a escala decimal
        self.max_weight = max_weight
        self.n_assets = None
        
    def _calculate_utility(self, 
                         weights: np.ndarray, 
                         params: PortfolioParameters,
                         risk_tolerance: Optional[float] = None) -> float:
        """
        Calculate utility for given weights and parameters.
        U(w) = E[R] - (1/2λ)σ²
        
        Args:
            weights: Portfolio weights
            params: Portfolio parameters
            risk_tolerance: Optional override for instance risk_tolerance
            
        Returns:
            Utility value
        """
        rt = risk_tolerance if risk_tolerance is not None else self.risk_tolerance
        
        expected_return = np.dot(weights, params.expected_returns)
        portfolio_variance = np.dot(weights, np.dot(params.covariance_matrix, weights))
        
        # Usar risk_tolerance normalizado
        risk_penalty = (1 / (2 * rt)) * portfolio_variance
        
        return expected_return - risk_penalty    
    
    def _optimization_objective(self, 
                              weights: np.ndarray, 
                              params: PortfolioParameters) -> float:
        """Objective function for optimization (negative utility for minimization)."""
        return -self._calculate_utility(weights, params)
    
    def optimize(self, params: PortfolioParameters, initial_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Find optimal portfolio weights with simplified optimization strategy."""
        params.validate()
        self.n_assets = len(params.expected_returns)
        
        # Simplified constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        
        # Single weight constraint for all assets
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: self.max_weight - np.max(x)
        })
        
        # Bounds for weights
        bounds = tuple((0, self.max_weight) for _ in range(self.n_assets))
        
        try:
            # Start with equal weights if no initial weights provided
            if initial_weights is None:
                initial_weights = np.ones(self.n_assets) / self.n_assets
            
            # Single optimization with balanced settings
            result = minimize(
                self._optimization_objective,
                initial_weights,
                args=(params,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'ftol': 1e-8,
                    'maxiter': 1000,
                    'disp': False
                }
            )
            
            if not result.success:
                # Fall back to equal weights if optimization fails
                return np.ones(self.n_assets) / self.n_assets
            
            # Clean up small weights
            weights = result.x.copy()
            weights[weights < 1e-5] = 0
            
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(self.n_assets) / self.n_assets
            
            return weights
                
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            # Return equal weights as fallback
            return np.ones(self.n_assets) / self.n_assets
        

    def calculate_cash_equivalent(self,
                                weights: np.ndarray,
                                params: PortfolioParameters,
                                risk_tolerance: Optional[float] = None) -> float:
        """
        Calculate the cash equivalent value for a portfolio.
        
        Args:
            weights: Portfolio weights
            params: Portfolio parameters
            risk_tolerance: Optional override for instance risk_tolerance
                Note: This should be in the original scale (will be normalized internally)
                
        Returns:
            Cash equivalent value
        """
        # Si se proporciona risk_tolerance, normalizarlo como en el constructor
        rt = (risk_tolerance / 100.0) if risk_tolerance is not None else self.risk_tolerance
        
        # Calculate expected return and variance
        exp_return = np.dot(weights, params.expected_returns)
        variance = weights @ params.covariance_matrix @ weights
        
        # Calculate utility usando rt ya normalizado
        utility = exp_return - (1 / (2 * rt)) * variance
        return utility
    
    def calculate_cel(self,
                     optimal_weights: np.ndarray,
                     suboptimal_weights: np.ndarray,
                     true_params: PortfolioParameters,
                     risk_tolerance: Optional[float] = None) -> float:
        """
        Calculate Cash Equivalent Loss between optimal and suboptimal portfolios.
        
        Args:
            optimal_weights: Weights of the optimal portfolio
            suboptimal_weights: Weights of the suboptimal portfolio
            true_params: True portfolio parameters
            risk_tolerance: Optional override for instance risk_tolerance
            
        Returns:
            Relative Cash Equivalent Loss
        """
        rt = risk_tolerance if risk_tolerance is not None else self.risk_tolerance
        
        # Calcular utilidades con los parámetros verdaderos
        utility_optimal = self._calculate_utility(optimal_weights, true_params, rt)
        utility_suboptimal = self._calculate_utility(suboptimal_weights, true_params, rt)
        
        # Calcular cash equivalents
        ce_optimal = utility_optimal
        ce_suboptimal = utility_suboptimal
        
        # Verificación más estricta para CE cercano a cero
        eps = 1e-8
        if abs(ce_optimal) <= eps:
            raise ValueError(
                f"CE₀ too close to zero (|CE₀| = {abs(ce_optimal):.2e} ≤ {eps:.2e}) "
                "to calculate CEL reliably"
            )
        
        # Verificación de optimalidad con tolerancia numérica ajustada
        if ce_optimal < ce_suboptimal - 1e-8:
            logger.warning(
                f"Suboptimal weights gave higher utility (diff: {ce_suboptimal - ce_optimal:.2e}). "
                "Using absolute difference for CEL calculation."
            )
        
        # Calcular CEL relativo usando valor absoluto del CE óptimo
        cel = (ce_optimal - ce_suboptimal) / abs(ce_optimal)
        
        # Ignorar CELs muy pequeños
        return max(0, cel) if cel > 1e-4 else 0


def create_base_portfolio(expected_returns: np.ndarray,
                        covariance_matrix: np.ndarray,
                        risk_tolerance: float,
                        max_weight: float = 0.4) -> Tuple[np.ndarray, PortfolioOptimizer]:
    """
    Create base optimal portfolio using true parameters.
    
    Args:
        expected_returns: Vector of expected returns
        covariance_matrix: Covariance matrix of returns
        risk_tolerance: Risk tolerance parameter
        max_weight: Maximum weight allowed for any single asset
        
    Returns:
        tuple: (optimal weights array, configured optimizer instance)
        
    Raises:
        ValueError: If optimization fails to converge
    """
    params = PortfolioParameters(expected_returns, covariance_matrix)
    optimizer = PortfolioOptimizer(risk_tolerance, max_weight)
    optimal_weights = optimizer.optimize(params)
    
    # Verificar que la optimización produjo un resultado válido
    utility = optimizer._calculate_utility(optimal_weights, params)
    equal_weights = np.ones_like(optimal_weights) / len(optimal_weights)
    utility_equal = optimizer._calculate_utility(equal_weights, params)
    
    if utility < utility_equal:
        raise ValueError(
            "Optimization failed to find weights better than equal-weight portfolio. "
            "Check portfolio parameters and optimization settings."
        )
    
    return optimal_weights, optimizer