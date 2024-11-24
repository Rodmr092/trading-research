# main.py
"""Main execution script for portfolio sensitivity analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from dataclasses import dataclass
from typing import Dict, List

# Import from local modules
from config import PortfolioConfig, ExperimentConfig
from experiment import run_single_trial
from visualizer import ResultVisualizer

def setup_environment():
    """Configure visualization and logging settings."""
    np.random.seed(42)
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_analysis(portfolio_config: PortfolioConfig, 
                experiment_config: ExperimentConfig):
    """Run complete sensitivity analysis."""
    setup_environment()
    experiment_config.validate()
    
    results = {}
    for rt in experiment_config.risk_tolerances:
        logging.info(f"Running experiment for risk tolerance {rt}")
        trial_results = []
        
        for _ in range(experiment_config.n_trials):
            cel_means, cel_vars, cel_covs = run_single_trial(
                portfolio_config, experiment_config.error_size, rt
            )
            trial_results.append({
                'means': cel_means,
                'variances': cel_vars,
                'covariances': cel_covs
            })
        
        results[rt] = pd.DataFrame(trial_results)
        logging.info(f"Completed experiment for RT={rt}")
    
    return results

# Data initialization
BASE_CORRELATION_MATRIX = np.array([
    [1.0000, 0.3660, 0.3457, 0.1606, 0.2279, 0.5133, 0.5203, 0.2176, 0.3267, 0.5101],
    [0.3660, 1.0000, 0.5379, 0.2165, 0.4986, 0.5823, 0.3569, 0.4760, 0.6517, 0.5853],
    [0.3457, 0.5379, 1.0000, 0.2218, 0.4283, 0.4051, 0.4492, 0.3867, 0.4883, 0.6569],
    [0.1606, 0.2165, 0.2218, 1.0000, 0.0569, 0.3609, 0.2325, 0.2289, 0.1726, 0.3814],
    [0.2279, 0.4986, 0.4283, 0.0569, 1.0000, 0.3619, 0.4483, 0.5952, 0.4378, 0.4368],
    [0.5133, 0.5823, 0.4051, 0.3609, 0.3619, 1.0000, 0.6167, 0.4996, 0.5811, 0.5644],
    [0.5203, 0.3569, 0.4492, 0.2325, 0.4483, 0.6167, 1.0000, 0.6037, 0.5671, 0.6032],
    [0.2176, 0.4760, 0.3867, 0.2289, 0.5952, 0.4996, 0.6037, 1.0000, 0.5012, 0.4772],
    [0.3267, 0.6517, 0.4883, 0.1726, 0.4378, 0.5811, 0.5671, 0.5012, 1.0000, 0.6039],
    [0.5101, 0.5853, 0.6569, 0.3814, 0.4368, 0.5644, 0.6032, 0.4772, 0.6039, 1.0000]
])

if __name__ == "__main__":
    # Initialize configurations
    portfolio_config = PortfolioConfig(
        n_assets=10,
        true_means=np.array([1.5617, 1.9477, 1.907, 1.5801, 2.1643, 
                            1.6010, 1.4892, 1.6248, 1.4075, 1.4537]) / 100,
        true_stds=np.array([8.8308, 8.4585, 10.040, 8.6215, 5.9886,
                           6.8767, 5.8162, 5.6385, 8.0047, 8.2125]) / 100,
        correlation_matrix=BASE_CORRELATION_MATRIX
    )
    
    experiment_config = ExperimentConfig(
        risk_tolerances=[25, 50, 75],
        error_size=0.10,
        n_trials=100
    )
    
    # Run analysis
    results = run_analysis(portfolio_config, experiment_config)
    
    # Visualize results
    visualizer = ResultVisualizer(results, experiment_config)
    visualizer.plot_distributions()
    visualizer.generate_summary_stats()