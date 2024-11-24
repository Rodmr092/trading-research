# experiment.py
"""Experiment execution module."""

import numpy as np
from optimizer import optimize_portfolio, negative_utility
from config import PortfolioConfig

def generate_error_matrix(true_matrix: np.ndarray, error_size: float, 
                         is_diagonal: bool = False) -> np.ndarray:
    """Generate matrix with random errors."""
    if is_diagonal:
        error = np.diag(true_matrix) * (1 + error_size * 
                                       np.random.randn(true_matrix.shape[0]))
        result = true_matrix.copy()
        np.fill_diagonal(result, error)
        return result
    
    error = true_matrix * (1 + error_size * np.random.randn(*true_matrix.shape))
    return error

def calculate_cel(true_weights: np.ndarray, error_weights: np.ndarray, 
                 means: np.ndarray, cov: np.ndarray, 
                 risk_tolerance: float) -> float:
    """Compute cash equivalent loss between optimal and sub-optimal portfolios."""
    true_utility = -negative_utility(true_weights, means, cov, risk_tolerance)
    error_utility = -negative_utility(error_weights, means, cov, risk_tolerance)
    return (true_utility - error_utility) / true_utility * 100

def run_single_trial(portfolio_config: PortfolioConfig, error_size: float, 
                    risk_tolerance: float) -> tuple:
    """Execute a single simulation trial."""
    true_means = portfolio_config.true_means
    true_cov = portfolio_config.covariance_matrix
    
    # Base portfolio
    base_weights = optimize_portfolio(true_means, true_cov, risk_tolerance)
    
    # Mean errors
    error_means = generate_error_matrix(true_means, error_size)
    weights_mean_error = optimize_portfolio(error_means, true_cov, risk_tolerance)
    cel_means = calculate_cel(base_weights, weights_mean_error, true_means, 
                            true_cov, risk_tolerance)
    
    # Variance errors
    error_var_cov = true_cov.copy()
    np.fill_diagonal(error_var_cov, 
                    np.diag(true_cov) * (1 + error_size * 
                                        np.random.randn(true_cov.shape[0])))
    weights_var_error = optimize_portfolio(true_means, error_var_cov, risk_tolerance)
    cel_vars = calculate_cel(base_weights, weights_var_error, true_means, 
                           true_cov, risk_tolerance)
    
    # Covariance errors
    error_cov_matrix = true_cov.copy()
    mask = ~np.eye(true_cov.shape[0], dtype=bool)
    error_cov_matrix[mask] *= (1 + error_size * np.random.randn(np.sum(mask)))
    weights_cov_error = optimize_portfolio(true_means, error_cov_matrix, risk_tolerance)
    cel_covs = calculate_cel(base_weights, weights_cov_error, true_means, 
                           true_cov, risk_tolerance)
    
    return cel_means, cel_vars, cel_covs

