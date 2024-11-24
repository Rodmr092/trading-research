# config.py
"""Configuration module for portfolio sensitivity analysis."""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import logging

@dataclass
class PortfolioConfig:
    """Portfolio configuration and market data."""
    n_assets: int
    true_means: np.ndarray
    true_stds: np.ndarray
    correlation_matrix: np.ndarray
    
    @property
    def covariance_matrix(self) -> np.ndarray:
        """Calculate the covariance matrix from std devs and correlations."""
        return np.outer(self.true_stds, self.true_stds) * self.correlation_matrix

@dataclass
class ExperimentConfig:
    """Configuration for sensitivity analysis experiments."""
    risk_tolerances: List[int]
    error_size: float
    n_trials: int = 100
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.risk_tolerances:
            raise ValueError("Must specify at least one risk tolerance")
        if not all(rt > 0 for rt in self.risk_tolerances):
            raise ValueError("All risk tolerances must be positive")
        if not 0 < self.error_size <= 1:
            raise ValueError("error_size must be between 0 and 1")
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
