# extended_experiment.py
"""Extended experiment module for additional portfolio sensitivity analyses."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
from config import PortfolioConfig, ExperimentConfig

@dataclass
class ExtendedExperimentConfig(ExperimentConfig):
    """Extended configuration for additional sensitivity analyses."""
    error_sizes: List[float]  # Multiple error sizes to test
    asset_counts: List[int]   # Different portfolio sizes to test
    allow_negative: bool = False  # Whether to allow negative weights
    
    def validate(self) -> None:
        """Validate extended configuration parameters."""
        super().validate()
        if not self.error_sizes:
            raise ValueError("Must specify at least one error size")
        if not all(0 < es <= 1 for es in self.error_sizes):
            raise ValueError("All error sizes must be between 0 and 1")
        if not self.asset_counts:
            raise ValueError("Must specify at least one asset count")
        if not all(ac > 1 for ac in self.asset_counts):
            raise ValueError("All asset counts must be greater than 1")

def optimize_portfolio_extended(means: np.ndarray, cov: np.ndarray, 
                             risk_tolerance: float, 
                             allow_negative: bool = False) -> np.ndarray:
    """Optimize portfolio with configurable constraints."""
    n = len(means)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Set bounds based on whether negative weights are allowed
    bounds = tuple((None, None) if allow_negative else (0, 1) for _ in range(n))
    
    result = minimize(
        lambda w: -(w @ means - (1/(2*risk_tolerance)) * (w @ cov @ w)),
        x0=np.ones(n)/n,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    return result.x

def analyze_error_sensitivity(portfolio_config: PortfolioConfig,
                            experiment_config: ExtendedExperimentConfig) -> pd.DataFrame:
    """Analyze sensitivity across different error sizes."""
    results = []
    
    for error_size in experiment_config.error_sizes:
        for rt in experiment_config.risk_tolerances:
            for _ in range(experiment_config.n_trials):
                cel_means, cel_vars, cel_covs = run_single_trial(
                    portfolio_config, error_size, rt, 
                    experiment_config.allow_negative
                )
                results.append({
                    'error_size': error_size,
                    'risk_tolerance': rt,
                    'cel_means': cel_means,
                    'cel_vars': cel_vars,
                    'cel_covs': cel_covs
                })
    
    return pd.DataFrame(results)

def analyze_portfolio_size(base_config: PortfolioConfig,
                         experiment_config: ExtendedExperimentConfig) -> pd.DataFrame:
    """Analyze impact of portfolio size on sensitivity."""
    results = []
    
    for n_assets in experiment_config.asset_counts:
        # Create reduced portfolio configuration
        reduced_config = PortfolioConfig(
            n_assets=n_assets,
            true_means=base_config.true_means[:n_assets],
            true_stds=base_config.true_stds[:n_assets],
            correlation_matrix=base_config.correlation_matrix[:n_assets, :n_assets]
        )
        
        for rt in experiment_config.risk_tolerances:
            for _ in range(experiment_config.n_trials):
                cel_means, cel_vars, cel_covs = run_single_trial(
                    reduced_config, 
                    experiment_config.error_sizes[0],  # Use first error size
                    rt,
                    experiment_config.allow_negative
                )
                results.append({
                    'n_assets': n_assets,
                    'risk_tolerance': rt,
                    'cel_means': cel_means,
                    'cel_vars': cel_vars,
                    'cel_covs': cel_covs
                })
    
    return pd.DataFrame(results)

def analyze_solution_stability(portfolio_config: PortfolioConfig,
                            experiment_config: ExtendedExperimentConfig,
                            n_perturbations: int = 100) -> pd.DataFrame:
    """Analyze stability of portfolio solutions under small perturbations."""
    results = []
    base_error = min(experiment_config.error_sizes) / 10  # Use small perturbation
    
    for rt in experiment_config.risk_tolerances:
        # Calculate base portfolio
        true_means = portfolio_config.true_means
        true_cov = portfolio_config.covariance_matrix
        base_weights = optimize_portfolio_extended(
            true_means, true_cov, rt, experiment_config.allow_negative
        )
        
        # Generate perturbed solutions
        weight_variations = []
        for _ in range(n_perturbations):
            perturbed_means = true_means * (1 + base_error * np.random.randn(len(true_means)))
            perturbed_weights = optimize_portfolio_extended(
                perturbed_means, true_cov, rt, experiment_config.allow_negative
            )
            weight_variations.append(perturbed_weights)
        
        # Calculate stability metrics
        weight_variations = np.array(weight_variations)
        mean_weights = np.mean(weight_variations, axis=0)
        std_weights = np.std(weight_variations, axis=0)
        
        results.append({
            'risk_tolerance': rt,
            'mean_deviation': np.mean(np.abs(weight_variations - base_weights)),
            'max_deviation': np.max(np.abs(weight_variations - base_weights)),
            'weight_std': np.mean(std_weights),
            'turnover_ratio': np.mean(np.abs(np.diff(weight_variations, axis=0)))
        })
    
    return pd.DataFrame(results)

