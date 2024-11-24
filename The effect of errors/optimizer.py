
# optimizer.py
"""Portfolio optimization module."""
import numpy as np
from scipy.optimize import minimize

def negative_utility(weights: np.ndarray, means: np.ndarray, 
                    cov: np.ndarray, risk_tolerance: float) -> float:
    """Calculate negative utility for portfolio optimization."""
    port_return = np.sum(weights * means)
    port_var = weights.T @ cov @ weights
    return -port_return + (1/(2*risk_tolerance)) * port_var

def optimize_portfolio(means: np.ndarray, cov: np.ndarray, 
                      risk_tolerance: float) -> np.ndarray:
    """Optimize portfolio weights using mean-variance utility."""
    n = len(means)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n))
    
    result = minimize(
        negative_utility,
        x0=np.ones(n)/n,
        args=(means, cov, risk_tolerance),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x

