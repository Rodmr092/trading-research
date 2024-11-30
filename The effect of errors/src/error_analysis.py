"""
Error Analysis Module for Portfolio Optimization.
Implements Monte Carlo simulation with progress tracking.
"""

import numpy as np
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from tqdm.notebook import tqdm
from src.portfolio_optimizer import (
    PortfolioParameters,
    PortfolioOptimizer
)

# Configure logging with more informative messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ErrorAnalysisConfig:
    """Configuration parameters for error analysis"""
    n_iterations: int = 50
    error_magnitudes: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.15, 0.20])
    )
    risk_tolerances: np.ndarray = field(
        default_factory=lambda: np.array([25, 75])
    )
    n_jobs: int = -1
    random_seed: Optional[int] = None
    batch_size: int = 100
    show_progress: bool = True  # Nuevo parámetro para controlar la visualización

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        
        if np.any(self.error_magnitudes <= 0):
            raise ValueError("error_magnitudes must be positive")
        
        if np.any(self.risk_tolerances <= 0):
            raise ValueError("risk_tolerances must be positive")
        
        if self.n_jobs < 0:
            self.n_jobs = min(os.cpu_count() or 1, 8)
        elif self.n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")

    def total_simulations(self) -> int:
        """Calculate total number of simulations."""
        return (len(['means', 'variances', 'covariances']) * 
                len(self.error_magnitudes) * 
                len(self.risk_tolerances) * 
                self.n_iterations)

class SimulationProgress:
    """Clase para manejar el progreso de la simulación."""
    def __init__(self, total_simulations: int, show_progress: bool = True):
        self.show_progress = show_progress
        if show_progress:
            self.pbar = tqdm(total=total_simulations, 
                           desc="Running simulations",
                           unit="sim")
        self.results_count = 0
        self.error_count = 0
        self.start_time = time.time()

    def update(self, n: int = 1, success: bool = True):
        """Update progress bar and counters."""
        if success:
            self.results_count += n
        else:
            self.error_count += n
        if self.show_progress:
            self.pbar.update(n)

    def get_stats(self) -> Dict:
        """Get current statistics."""
        elapsed = time.time() - self.start_time
        return {
            'successful_simulations': self.results_count,
            'failed_simulations': self.error_count,
            'elapsed_time': elapsed,
            'simulations_per_second': (self.results_count + self.error_count) / elapsed if elapsed > 0 else 0
        }

    def close(self):
        """Close progress bar and log final statistics."""
        if self.show_progress:
            self.pbar.close()
        stats = self.get_stats()
        logger.info(f"""
Simulation completed:
- Successful simulations: {stats['successful_simulations']:,}
- Failed simulations: {stats['failed_simulations']:,}
- Total time: {stats['elapsed_time']:.1f} seconds
- Average speed: {stats['simulations_per_second']:.1f} sim/s
""")

class ErrorAnalyzer:
    """Implements Monte Carlo simulation with progress tracking."""
    
    def __init__(self, config: Optional[ErrorAnalysisConfig] = None):
        self.config = config or ErrorAnalysisConfig()
        self.base_results = {}
        self.rng = np.random.RandomState(self.config.random_seed)
        
    def _generate_error_params(self, 
                            error_type: str,
                            true_params: PortfolioParameters,
                            error_magnitude: float,
                            batch_size: int,
                            seed: Optional[int] = None) -> List[PortfolioParameters]:
        """Generate multiple error parameters in batch."""
        rng = np.random.RandomState(seed)
        error_params_list = []
        
        try:
            for _ in range(batch_size):
                if error_type == 'means':
                    z = rng.standard_normal(len(true_params.expected_returns))
                    returns = true_params.expected_returns * (1 + error_magnitude * 1.5 * z)
                    error_params = PortfolioParameters(
                        expected_returns=returns,
                        covariance_matrix=true_params.covariance_matrix.copy()
                    )
                elif error_type == 'variances':
                    n_assets = len(true_params.expected_returns)
                    variances = np.diag(true_params.covariance_matrix).copy()
                    std_devs = np.sqrt(variances)
                    corr_matrix = true_params.covariance_matrix / np.outer(std_devs, std_devs)
                    
                    z = rng.standard_normal(n_assets)
                    new_variances = variances * (1 + error_magnitude * z)
                    new_std_devs = np.sqrt(new_variances)
                    
                    cov_matrix = corr_matrix * np.outer(new_std_devs, new_std_devs)
                    cov_matrix = (cov_matrix + cov_matrix.T) / 2
                    
                    error_params = PortfolioParameters(
                        expected_returns=true_params.expected_returns.copy(),
                        covariance_matrix=cov_matrix
                    )
                else:  # covariances
                    n_assets = len(true_params.expected_returns)
                    variances = np.diag(true_params.covariance_matrix).copy()
                    std_devs = np.sqrt(variances)
                    corr_matrix = true_params.covariance_matrix / np.outer(std_devs, std_devs)
                    
                    z = rng.standard_normal((n_assets, n_assets))
                    z = (z + z.T) / 2
                    np.fill_diagonal(z, 0)
                    
                    new_corr = corr_matrix * (1 + error_magnitude * z)
                    np.fill_diagonal(new_corr, 1)
                    new_corr = (new_corr + new_corr.T) / 2
                    
                    cov_matrix = new_corr * np.outer(std_devs, std_devs)
                    error_params = PortfolioParameters(
                        expected_returns=true_params.expected_returns.copy(),
                        covariance_matrix=cov_matrix
                    )
                    
                error_params_list.append(error_params)
                
            return error_params_list
        except Exception as e:
            logger.error(f"Error generating parameters for {error_type}: {str(e)}")
            return []  # Retornar lista vacía en lugar de None

    def _run_batch_simulation(self, args: Tuple[str, float, float, PortfolioParameters, int, int]) -> Tuple[List[Dict], int, int]:
        """Run a batch of simulations with error handling."""
        error_type, error_magnitude, risk_tolerance, true_params, seed, batch_size = args
        results = []
        success_count = 0
        failure_count = 0
        
        try:
            error_params_list = self._generate_error_params(
                error_type, true_params, error_magnitude, batch_size, seed
            )
            
            if not error_params_list:
                return [], 0, batch_size
            
            optimizer = PortfolioOptimizer(risk_tolerance)
            optimal_weights = optimizer.optimize(true_params)
            
            if optimal_weights is None:
                return [], 0, batch_size
                
            for error_params in error_params_list:
                try:
                    suboptimal_weights = optimizer.optimize(error_params)
                    
                    if suboptimal_weights is None:
                        failure_count += 1
                        continue
                    
                    cel = optimizer.calculate_cel(
                        optimal_weights,
                        suboptimal_weights,
                        true_params,
                        risk_tolerance
                    )
                    
                    if cel is not None and not np.isnan(cel):
                        weight_diff = np.abs(optimal_weights - suboptimal_weights)
                        result = {
                            'error_type': error_type,
                            'error_magnitude': error_magnitude,
                            'risk_tolerance': risk_tolerance,
                            'cel': cel,
                            'max_weight_diff': np.max(weight_diff),
                            'mean_weight_diff': np.mean(weight_diff),
                            'active_positions': np.sum(optimal_weights > 0.01)
                        }
                        results.append(result)
                        success_count += 1
                    else:
                        failure_count += 1
                        
                except Exception as e:
                    logger.debug(f"Error in individual simulation: {str(e)}")
                    failure_count += 1
                        
        except Exception as e:
            logger.error(f"Error in batch simulation: {str(e)}")
            return [], 0, batch_size
        
        return results, success_count, failure_count

    def analyze_errors(self, true_params: PortfolioParameters) -> pd.DataFrame:
        """Run complete error analysis with progress tracking."""
        true_params.validate()
        
        # Initialize progress tracking
        total_sims = self.config.total_simulations()
        progress = SimulationProgress(total_sims, self.config.show_progress)
        
        logger.info(f"""
Starting error analysis:
- Total simulations: {total_sims:,}
- Parallel jobs: {self.config.n_jobs}
- Batch size: {self.config.batch_size}
""")
        
        # Prepare simulation parameters
        all_params = []
        base_seed = self.config.random_seed if self.config.random_seed is not None else 42
        
        for error_type in ['means', 'variances', 'covariances']:
            for error_magnitude in self.config.error_magnitudes:
                for risk_tolerance in self.config.risk_tolerances:
                    remaining_iterations = self.config.n_iterations
                    batch_counter = 0
                    
                    while remaining_iterations > 0:
                        batch_size = min(remaining_iterations, self.config.batch_size)
                        seed = base_seed + batch_counter if base_seed is not None else None
                        
                        all_params.append(
                            (error_type, error_magnitude, risk_tolerance, 
                             true_params, seed, batch_size)
                        )
                        
                        remaining_iterations -= batch_size
                        batch_counter += 1
        
        # Run simulations in parallel with progress tracking
        all_results = []
        with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
            for batch_results, successes, failures in executor.map(self._run_batch_simulation, all_params):
                if batch_results:
                    all_results.extend(batch_results)
                progress.update(successes, True)
                if failures > 0:
                    progress.update(failures, False)
        
        progress.close()
        
        if not all_results:
            raise ValueError("No valid results obtained from simulations")
        
        # Convert results to DataFrame and calculate statistics
        results_df = pd.DataFrame(all_results)
        stats = results_df.groupby(['error_type', 'error_magnitude', 'risk_tolerance']).agg({
            'cel': ['mean', 'std', 'min', 'max'],
            'max_weight_diff': ['mean', 'max'],
            'mean_weight_diff': 'mean',
            'active_positions': 'mean'
        }).round(4)
        
        logger.info("\nAnalysis completed successfully. Computing final statistics...")
        
        return stats

def run_error_analysis(expected_returns: np.ndarray,
                      covariance_matrix: np.ndarray,
                      config: Optional[ErrorAnalysisConfig] = None) -> pd.DataFrame:
    """Convenience function to run complete error analysis with progress tracking."""
    true_params = PortfolioParameters(expected_returns, covariance_matrix)
    analyzer = ErrorAnalyzer(config)
    return analyzer.analyze_errors(true_params)