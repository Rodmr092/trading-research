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
            
        self.risk_tolerance = risk_tolerance
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
        
        # Escalar el riesgo para hacerlo más comparable con el retorno
        risk_penalty = (1 / (2 * rt)) * portfolio_variance
        
        return expected_return - risk_penalty
    
    def _optimization_objective(self, 
                              weights: np.ndarray, 
                              params: PortfolioParameters) -> float:
        """Objective function for optimization (negative utility for minimization)."""
        return -self._calculate_utility(weights, params)
    
    def optimize(self, params: PortfolioParameters, initial_weights: Optional[np.ndarray] = None) -> np.ndarray:
        params.validate()
        self.n_assets = len(params.expected_returns)  # Mover esto al inicio
        start_time = time.perf_counter()
        logger.info(f"Starting optimization for {self.n_assets} assets")
        """
        Find optimal portfolio weights by maximizing utility with diversification constraints.
        
        Args:
            params: Portfolio parameters
            initial_weights: Initial guess for optimization
            
        Returns:
            Optimal portfolio weights
            
        Raises:
            ValueError: If optimization fails to converge after all attempts
        """
        params.validate()
        
        # Constraints for all optimization attempts
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        
        # Weight constraints usando closure para evitar problemas de late binding
        def make_constraint(idx):
            return lambda x: self.max_weight - x[idx]
        
        for i in range(self.n_assets):
            constraints.append({
                'type': 'ineq',
                'fun': make_constraint(i)
            })
        
        # Bounds for weights (no short-selling)
        bounds = tuple((0, self.max_weight) for _ in range(self.n_assets))
        
        # Función helper para evaluar la calidad de una solución
        def evaluate_solution(weights):
            if not np.all(np.isfinite(weights)):
                return float('-inf')
            if not (np.all(weights >= -1e-10) and np.all(weights <= self.max_weight + 1e-10)):
                return float('-inf')
            if not np.isclose(np.sum(weights), 1.0, rtol=1e-10, atol=1e-10):
                return float('-inf')
            return -self._optimization_objective(weights, params)
        
        try:
            best_result = None
            best_utility = float('-inf')
            
            # Generar múltiples puntos iniciales
            initial_points = []
            
            # 1. Usar initial_weights si se proporcionan
            if initial_weights is not None:
                initial_points.append(initial_weights)
            
            # 2. Equal weights
            initial_points.append(np.ones(self.n_assets) / self.n_assets)
            
            # 3. Random weights (varios intentos)
            np.random.seed(42)  # Para reproducibilidad
            for _ in range(3):
                weights = np.random.dirichlet(np.ones(self.n_assets))
                # Ajustar para respetar max_weight
                if np.any(weights > self.max_weight):
                    weights = np.minimum(weights, self.max_weight)
                    weights = weights / np.sum(weights)
                initial_points.append(weights)
            
            # 4. Concentrated portfolios
            for i in range(self.n_assets):
                weights = np.zeros(self.n_assets)
                weights[i] = self.max_weight
                remaining = 1.0 - self.max_weight
                other_weights = remaining / (self.n_assets - 1)
                weights[weights == 0] = other_weights
                initial_points.append(weights)
            
            # Diferentes configuraciones de optimización
            optimization_configs = [
                {'ftol': 1e-12, 'maxiter': 2000},  # Estricto
                {'ftol': 1e-9, 'maxiter': 3000},   # Balance
                {'ftol': 1e-6, 'maxiter': 5000},   # Relajado
            ]
            
            for init_point in initial_points:
                for config in optimization_configs:
                    try:
                        result = minimize(
                            self._optimization_objective,
                            init_point,
                            args=(params,),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={**config, 'disp': False}
                        )
                        
                        # Evaluar la solución solo si la optimización convergió
                        if result.success:
                            utility = evaluate_solution(result.x)
                            if utility > best_utility:
                                best_utility = utility
                                best_result = result
                                
                    except Exception as e:
                        logger.warning(f"Optimization attempt failed: {str(e)}")
                        continue
            
            if best_result is None:
                raise ValueError("Optimization failed to converge from any starting point or configuration")
            
            # Clean up small weights and renormalize
            weights = best_result.x.copy()
            weights[weights < 1e-4] = 0
            
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                logger.warning("All weights were eliminated in cleanup. Using equal weights.")
                weights = np.ones(self.n_assets) / self.n_assets
            
            # Verificación final
            final_utility = evaluate_solution(weights)
            equal_weights_utility = evaluate_solution(np.ones(self.n_assets) / self.n_assets)
            
            if final_utility <= equal_weights_utility:
                logger.warning("Optimization result worse than equal weights. Using equal weights.")
                weights = np.ones(self.n_assets) / self.n_assets
            
            return weights
                
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            raise
        for i, init_point in enumerate(initial_points):
            for j, config in enumerate(optimization_configs):
                try:
                    iter_start = time.perf_counter()
                    result = minimize(
                        self._optimization_objective,
                        init_point,
                        args=(params,),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={**config, 'disp': False}
                    )
                    
                    if result.success:
                        utility = evaluate_solution(result.x)
                        logger.debug(f"Attempt {i}-{j} succeeded: utility={utility:.6f}, "
                                f"time={time.perf_counter()-iter_start:.2f}s")
                        if utility > best_utility:
                            best_utility = utility
                            best_result = result
                            
                except Exception as e:
                    logger.warning(f"Optimization attempt {i}-{j} failed: {str(e)}")
        
        logger.info(f"Optimization completed in {time.perf_counter()-start_time:.2f}s")

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
            
        Returns:
            Cash equivalent value
        """
        rt = risk_tolerance if risk_tolerance is not None else self.risk_tolerance
        
        # Calculate expected return and variance
        exp_return = np.dot(weights, params.expected_returns)
        variance = weights @ params.covariance_matrix @ weights
        
        # Calculate utility
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
            
        Raises:
            ValueError: If CE₀ too close to zero or if optimal weights give lower utility
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
        
        # Verificación de optimalidad con tolerancia numérica
        if ce_optimal < ce_suboptimal - 1e-10:
            logger.warning(
                f"Suboptimal weights gave higher utility (diff: {ce_suboptimal - ce_optimal:.2e}). "
                "Using absolute difference for CEL calculation."
            )
        
        # Calcular CEL relativo usando valor absoluto del CE óptimo
        cel = (ce_optimal - ce_suboptimal) / abs(ce_optimal)
        
        return max(0, cel)  # CEL no debería ser negativo


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