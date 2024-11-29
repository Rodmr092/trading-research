"""
Integration test script for portfolio optimization error analysis.
Tests the complete workflow from data acquisition to error analysis.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data_management import DataManager
from src.portfolio_optimizer import PortfolioParameters
from src.error_analysis import ErrorAnalysisConfig, run_error_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_integration_test(
    symbols: list = None,
    risk_tolerances: list = None,
    error_magnitudes: list = None,
    n_iterations: int = 10  # Reducido a 10 para prueba inicial
) -> dict:
    """
    Run complete integration test of the portfolio optimization system.
    
    Args:
        symbols: List of stock symbols to analyze
        risk_tolerances: List of risk tolerance values
        error_magnitudes: List of error magnitude values
        n_iterations: Number of Monte Carlo iterations
        
    Returns:
        Dictionary containing test results and metrics
    """
    try:
        # Default values if not provided
        if symbols is None:
            symbols = [
                'MSFT', 'AAPL', 'NVDA', 'AMZN', 'META'  # Reducido a 5 símbolos
            ]
            
        if risk_tolerances is None:
            risk_tolerances = [25]  # Solo un valor para prueba inicial
            
        if error_magnitudes is None:
            error_magnitudes = [0.05]  # Solo un valor para prueba inicial
        
        logger.info("Starting integration test...")
        
        # 1. Data Management
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)
        
        data_manager = DataManager(
            symbols=symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Process data
        statistics = data_manager.process_all()
        
        if statistics is None or 'expected_returns' not in statistics:
            raise ValueError("Failed to obtain portfolio statistics")
            
        logger.info("Data processing completed successfully")
        
        # 2. Create Portfolio Parameters
        params = PortfolioParameters(
            expected_returns=statistics['expected_returns'],
            covariance_matrix=statistics['covariance_matrix']
        )
        
        # 3. Configure Error Analysis
        config = ErrorAnalysisConfig(
            n_iterations=n_iterations,
            error_magnitudes=np.array(error_magnitudes),
            risk_tolerances=np.array(risk_tolerances),
            random_seed=42
        )
        
        # 4. Run Error Analysis
        logger.info("Starting error analysis...")
        results = run_error_analysis(
            expected_returns=params.expected_returns,
            covariance_matrix=params.covariance_matrix,
            config=config
        )
        
        logger.info("Error analysis completed successfully")
        
        # 5. Prepare summary statistics
        summary = {
            'symbols': symbols,
            'data_period': f"{start_date.date()} to {end_date.date()}",
            'n_assets': len(symbols),
            'n_iterations': n_iterations,
            'risk_tolerances': risk_tolerances,
            'error_magnitudes': error_magnitudes,
            'results': results
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        raise

def print_test_results(summary: dict) -> None:
    """
    Print formatted test results.
    
    Args:
        summary: Dictionary containing test results
    """
    print("\n=== Integration Test Results ===")
    print(f"\nData Period: {summary['data_period']}")
    print(f"Assets analyzed: {len(summary['symbols'])}")
    print("\nSymbols:", ', '.join(summary['symbols']))
    
    print("\nError Analysis Results:")
    print("----------------------")
    
    # Print results in a more readable format
    results_df = summary['results']
    
    for error_type in ['means', 'variances', 'covariances']:
        print(f"\n{error_type.capitalize()} Errors:")
        print("-" * 80)
        
        subset = results_df.xs(error_type, level='error_type')
        print(subset.to_string())
        
    print("\nTest completed successfully!")

if __name__ == "__main__":
    try:
        # Run integration test
        test_results = run_integration_test()
        
        # Print results
        print_test_results(test_results)
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise