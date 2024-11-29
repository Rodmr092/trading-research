"""
Error Analysis Module for Portfolio Optimization.
Implements Monte Carlo simulation to analyze the impact of estimation errors
in means, variances, and covariances on portfolio performance.
"""

import numpy as np
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from src.portfolio_optimizer import (
    PortfolioParameters,
    PortfolioOptimizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ErrorAnalysisConfig:
    """Configuration parameters for error analysis"""
    n_iterations: int = 100  # Aumentado a 100
    error_magnitudes: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.10, 0.15, 0.20])
    )
    risk_tolerances: np.ndarray = field(
        default_factory=lambda: np.array([25, 50, 75])
    )
    n_jobs: int = -1
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        
        if np.any(self.error_magnitudes <= 0):
            raise ValueError("error_magnitudes must be positive")
        
        if np.any(self.risk_tolerances <= 0):
            raise ValueError("risk_tolerances must be positive")
        
        # Convertir -1 al número de CPUs disponibles
        if self.n_jobs < 0:
            self.n_jobs = os.cpu_count() or 1
        elif self.n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        
class ErrorAnalyzer:
    """
    Implements Monte Carlo simulation to analyze the impact of estimation errors
    on portfolio optimization results.
    """
    
    def __init__(self, config: Optional[ErrorAnalysisConfig] = None):
        """Initialize ErrorAnalyzer with configuration parameters."""
        if config is not None and config.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        self.config = config or ErrorAnalysisConfig()
        self.base_results = {}
        self.rng = np.random.RandomState(self.config.random_seed)

    def _check_matrix_properties(self,
                               cov_matrix: np.ndarray,
                               original_cov: np.ndarray,
                               error_type: str) -> Dict[str, float]:
        """
        Check statistical properties of the generated covariance matrix.
        
        Args:
            cov_matrix: Generated covariance matrix
            original_cov: Original covariance matrix
            error_type: Type of error being checked ('variance' or 'covariance')
            
        Returns:
            Dictionary with quality metrics
        """
        n_assets = len(cov_matrix)
        
        # Extraer matrices de correlación
        std_orig = np.sqrt(np.diag(original_cov))
        std_new = np.sqrt(np.diag(cov_matrix))
        corr_orig = original_cov / np.outer(std_orig, std_orig)
        corr_new = cov_matrix / np.outer(std_new, std_new)
        
        # Calcular métricas
        metrics = {
            'positive_definite': np.all(np.linalg.eigvals(cov_matrix) > -1e-10),
            'symmetric': np.allclose(cov_matrix, cov_matrix.T),
            'variance_ratio': np.mean(std_new**2 / std_orig**2),
            'correlation_difference': np.mean(np.abs(corr_new - corr_orig)),
            'max_correlation': np.max(np.abs(corr_new[~np.eye(n_assets, dtype=bool)]))
        }
        
        return metrics
        
    def _generate_mean_errors(self, 
                            true_params: PortfolioParameters,
                            error_magnitude: float,
                            seed: Optional[int] = None) -> PortfolioParameters:
        """
        Generate parameters with errors in means.
        μᵢ_error = μᵢ(1 + kzᵢ) where zᵢ ~ N(0,1)
        """
        rng = np.random.RandomState(seed)
        z = rng.standard_normal(len(true_params.expected_returns))
        
        # Aumentar ligeramente el impacto del error
        error_params = PortfolioParameters(
            expected_returns=true_params.expected_returns * (1 + error_magnitude * 1.5 * z),
            covariance_matrix=true_params.covariance_matrix.copy()
        )
        return error_params

    def _generate_variance_errors(self,
                                true_params: PortfolioParameters,
                                error_magnitude: float,
                                seed: Optional[int] = None) -> PortfolioParameters:
        """
        Generate parameters with errors in variances.
        σᵢᵢ_error = σᵢᵢ(1 + kzᵢ) where zᵢ ~ N(0,1)
        """
        rng = np.random.RandomState(seed)
        n_assets = len(true_params.expected_returns)
        
        # Extraer varianzas y matriz de correlación
        variances = np.diag(true_params.covariance_matrix).copy()
        std_devs = np.sqrt(variances)
        corr_matrix = true_params.covariance_matrix / np.outer(std_devs, std_devs)
        
        # Generar errores lineales para varianzas
        z = rng.standard_normal(n_assets)
        new_variances = variances * (1 + error_magnitude * z)  # Perturbación lineal
        new_std_devs = np.sqrt(new_variances)
        
        # Reconstruir matriz de covarianza manteniendo correlaciones
        cov_matrix = corr_matrix * np.outer(new_std_devs, new_std_devs)
        
        # Asegurar simetría y definición positiva
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        min_eigenval = np.min(np.linalg.eigvals(cov_matrix))
        if min_eigenval < 0:
            cov_matrix += (-min_eigenval + 1e-8) * np.eye(n_assets)
        
        error_params = PortfolioParameters(
            expected_returns=true_params.expected_returns.copy(),
            covariance_matrix=cov_matrix
        )
        return error_params

    def _generate_covariance_errors(self,
                                true_params: PortfolioParameters,
                                error_magnitude: float,
                                seed: Optional[int] = None) -> PortfolioParameters:
        """
        Generate parameters with errors in covariances.
        σᵢⱼ_error = σᵢⱼ(1 + kzᵢⱼ) where zᵢⱼ ~ N(0,1)
        """
        rng = np.random.RandomState(seed)
        n_assets = len(true_params.expected_returns)
        
        # Extraer varianzas y matriz de correlación
        variances = np.diag(true_params.covariance_matrix).copy()
        std_devs = np.sqrt(variances)
        corr_matrix = true_params.covariance_matrix / np.outer(std_devs, std_devs)
        
        # Generar errores para correlaciones
        z = rng.standard_normal((n_assets, n_assets))
        z = (z + z.T) / 2
        np.fill_diagonal(z, 0)
        
        # Aplicar perturbación lineal a las correlaciones
        new_corr = corr_matrix * (1 + error_magnitude * z)
        np.fill_diagonal(new_corr, 1)
        
        # Asegurar simetría
        new_corr = (new_corr + new_corr.T) / 2
        
        # Verificar y ajustar definición positiva
        eigvals, eigvecs = np.linalg.eigh(new_corr)
        if np.any(eigvals < 0):
            logger.debug(f"Adjusting negative eigenvalues: min={np.min(eigvals):.2e}")
            eigvals = np.maximum(eigvals, 1e-8)
            new_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Reconstruir matriz de covarianza
        cov_matrix = new_corr * np.outer(std_devs, std_devs)
        
        return PortfolioParameters(
            expected_returns=true_params.expected_returns.copy(),
            covariance_matrix=cov_matrix
        )
        
    def _run_single_simulation(self, args: Tuple[str, float, float, PortfolioParameters, int]) -> Dict:
        """Run a single iteration of the simulation."""
        error_type, error_magnitude, risk_tolerance, true_params, seed = args
        start_time = time.perf_counter()
        
        try:
            # Validaciones adicionales
            if np.any(np.isnan(true_params.covariance_matrix)) or np.any(np.isinf(true_params.covariance_matrix)):
                raise ValueError("Invalid values in covariance matrix")

            # Verificar definición positiva
            eigenvals = np.linalg.eigvals(true_params.covariance_matrix)
            if not np.all(eigenvals > 1e-8):
                raise ValueError("Covariance matrix is not positive definite")

            # Validar dimensiones de entrada
            if len(true_params.expected_returns) != true_params.covariance_matrix.shape[0]:
                raise ValueError("Dimension mismatch between returns and covariance matrix")
            
            # Reduced logging - solo para combinaciones importantes
            is_logging_iteration = (error_magnitude in [0.05, 0.20] and 
                                risk_tolerance == 50)
            
            if is_logging_iteration:
                logger.info(f"Starting simulation: {error_type}, k={error_magnitude}, rt={risk_tolerance}")

            # Generate parameters with errors based on type
            if error_type == 'means':
                error_params = self._generate_mean_errors(true_params, error_magnitude, seed)
                matrix_checks = None
                if is_logging_iteration:
                    rel_diff = np.abs((error_params.expected_returns - true_params.expected_returns) / 
                                    true_params.expected_returns)
                    logger.info(f"Mean relative difference in returns: {np.mean(rel_diff):.2%}")
            elif error_type == 'variances':
                error_params = self._generate_variance_errors(true_params, error_magnitude, seed)
                matrix_checks = self._check_matrix_properties(
                    error_params.covariance_matrix,
                    true_params.covariance_matrix,
                    'variance'
                )
                if is_logging_iteration:
                    var_diff = np.abs((np.diag(error_params.covariance_matrix) - 
                                    np.diag(true_params.covariance_matrix)) / 
                                    np.diag(true_params.covariance_matrix))
                    logger.info(f"Mean relative difference in variances: {np.mean(var_diff):.2%}")
            else:  # covariances
                error_params = self._generate_covariance_errors(true_params, error_magnitude, seed)
                matrix_checks = self._check_matrix_properties(
                    error_params.covariance_matrix,
                    true_params.covariance_matrix,
                    'covariance'
                )
                if is_logging_iteration:
                    mask = ~np.eye(true_params.covariance_matrix.shape[0], dtype=bool)
                    cov_diff = np.abs((error_params.covariance_matrix[mask] - 
                                    true_params.covariance_matrix[mask]) / 
                                    true_params.covariance_matrix[mask])
                    logger.info(f"Mean relative difference in covariances: {np.mean(cov_diff):.2%}")
                
            # Optimizar portafolios
            optimizer = PortfolioOptimizer(risk_tolerance)
            try:
                optimal_weights = optimizer.optimize(true_params)
                suboptimal_weights = optimizer.optimize(error_params)
                
                if optimal_weights is None or suboptimal_weights is None:
                    raise ValueError("Optimization failed to produce valid weights")
                
                # Calcular métricas adicionales
                optimal_return = np.dot(optimal_weights, true_params.expected_returns)
                optimal_risk = np.sqrt(optimal_weights @ true_params.covariance_matrix @ optimal_weights)
                subopt_return = np.dot(suboptimal_weights, true_params.expected_returns)
                subopt_risk = np.sqrt(suboptimal_weights @ true_params.covariance_matrix @ suboptimal_weights)
                
                weight_diff = np.abs(optimal_weights - suboptimal_weights)
                if is_logging_iteration:
                    logger.info(f"Weight differences: {np.mean(weight_diff):.4f}")
                    
                cel = optimizer.calculate_cel(
                    optimal_weights,
                    suboptimal_weights,
                    true_params,
                    risk_tolerance
                )
                
                if cel is None or np.isnan(cel):
                    raise ValueError("Invalid CEL calculation result")
                
                if is_logging_iteration:
                    logger.info(f"CEL: {cel:.4f}, Time: {time.perf_counter() - start_time:.2f}s")
                    
                result = {
                    'error_type': error_type,
                    'error_magnitude': error_magnitude,
                    'risk_tolerance': risk_tolerance,
                    'cel': cel,
                    'time': time.perf_counter() - start_time,
                    'optimal_return': optimal_return,
                    'optimal_risk': optimal_risk,
                    'suboptimal_return': subopt_return,
                    'suboptimal_risk': subopt_risk,
                    'max_weight_diff': np.max(weight_diff),
                    'mean_weight_diff': np.mean(weight_diff),
                    'active_positions': np.sum(optimal_weights > 0.01)
                }
                
                if matrix_checks:
                    result.update({f'matrix_{k}': v for k, v in matrix_checks.items()})
                
                return result
                    
            except Exception as e:
                logger.error(f"Optimization error: {str(e)}")
                return None
                    
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            return None
    
    def analyze_errors(self, true_params: PortfolioParameters) -> pd.DataFrame:
        """Run complete error analysis using Monte Carlo simulation.
        
        Args:
            true_params: Portfolio parameters to analyze
            
        Returns:
            DataFrame with error analysis results
            
        Raises:
            ValueError: If parameters are invalid or optimization fails
        """
        # Validaciones iniciales
        if np.any(np.isnan(true_params.covariance_matrix)) or np.any(np.isinf(true_params.covariance_matrix)):
            raise ValueError("Invalid values in covariance matrix")

        if len(true_params.expected_returns) != true_params.covariance_matrix.shape[0]:
            raise ValueError("Dimension mismatch between returns and covariance matrix")

        # Verificar matriz de covarianza
        try:
            eigenvals = np.linalg.eigvals(true_params.covariance_matrix)
            if not np.all(eigenvals > 1e-8):
                raise ValueError("Covariance matrix is not positive definite")
        except np.linalg.LinAlgError:
            raise ValueError("Invalid covariance matrix")

        results = []
        base_seed = self.config.random_seed if self.config.random_seed is not None else 42
        
        # Generate all simulation parameters with seeds
        simulation_params = []
        iteration_counter = 0
        total_combinations = (
            len(['means', 'variances', 'covariances']) *
            len(self.config.error_magnitudes) *
            len(self.config.risk_tolerances)
        )
        
        counter = 0
        for error_type in ['means', 'variances', 'covariances']:
            for error_magnitude in self.config.error_magnitudes:
                for risk_tolerance in self.config.risk_tolerances:
                    counter += 1
                    logger.info(
                        f"Processing combination {counter}/{total_combinations}: "
                        f"{error_type}, k={error_magnitude}, rt={risk_tolerance}"
                    )
                    for i in range(self.config.n_iterations):
                        seed = base_seed + iteration_counter
                        simulation_params.append(
                            (error_type, error_magnitude, risk_tolerance, true_params, seed)
                        )
                        iteration_counter += 1
        
        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
            simulation_results = list(executor.map(self._run_single_simulation, simulation_params))
        
        # Process results
        for result in simulation_results:
            if result is not None:
                results.append(result)
        
        if not results:
            raise ValueError("Optimization failed - no valid results obtained")
        
        # Convertir resultados a DataFrame
        results_df = pd.DataFrame(results)
        
        # Calcular estadísticas agregadas
        stats = results_df.groupby(['error_type', 'error_magnitude', 'risk_tolerance']).agg({
            'cel': ['mean', 'std', 'min', 'max'],
            'max_weight_diff': ['mean', 'max'],
            'mean_weight_diff': 'mean',
            'active_positions': 'mean',
            'optimal_return': 'mean',
            'optimal_risk': 'mean',
            'suboptimal_return': 'mean',
            'suboptimal_risk': 'mean',
            'time': 'mean'
        }).round(4)
        
        # Log resumen de resultados
        for (error_type, k, rt), group in results_df.groupby(['error_type', 'error_magnitude', 'risk_tolerance']):
            logger.info(f"\nResults for {error_type}, k={k}, rt={rt}:")
            logger.info(f"CEL: mean={group['cel'].mean():.4f}, std={group['cel'].std():.4f}")
            logger.info(f"Weight differences: mean={group['mean_weight_diff'].mean():.4f}")
            logger.info(f"Average processing time: {group['time'].mean():.2f}s")
        
        return stats

def run_error_analysis(expected_returns: np.ndarray,
                      covariance_matrix: np.ndarray,
                      config: Optional[ErrorAnalysisConfig] = None) -> pd.DataFrame:
    """
    Convenience function to run complete error analysis.
    """
    true_params = PortfolioParameters(expected_returns, covariance_matrix)
    analyzer = ErrorAnalyzer(config)
    
    # Mover el logging aquí
    logger.info(f"Starting analysis with {analyzer.config.n_iterations} iterations per combination")
    total_combinations = (
        len(['means', 'variances', 'covariances']) * 
        len(analyzer.config.error_magnitudes) * 
        len(analyzer.config.risk_tolerances)
    )
    logger.info(f"Total parameter combinations: {total_combinations}")
    total_sims = total_combinations * analyzer.config.n_iterations
    logger.info(f"Total simulations to run: {total_sims}")
    
    results = analyzer.analyze_errors(true_params)
    return results