"""
Data Management Module for Portfolio Optimization Analysis.
Handles data downloading, processing, and statistical calculations.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages the acquisition and processing of financial data for portfolio optimization.
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        data_dir: Union[str, Path] = "data"
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data attributes
        self.prices = None
        self.returns = None
        self.statistics = None
        
        logger.info(f"Initialized DataManager with {len(symbols)} symbols")

    def download_data(self) -> None:
        """Download and clean historical price data."""
        try:
            logger.info("Downloading historical data...")
            
            # Download data for all symbols
            data = yf.download(
                self.symbols,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                self.prices = data['Adj Close']
            else:
                self.prices = data  # Single symbol case
                self.prices.columns = [self.symbols[0]]
            
            # Handle missing values
            if self.prices.isnull().any().any():
                logger.warning("Missing values detected in price data")
                missing_pct = self.prices.isnull().mean()
                
                # Remove symbols with too many missing values
                valid_symbols = missing_pct[missing_pct < 0.1].index
                removed_symbols = set(self.symbols) - set(valid_symbols)
                
                if removed_symbols:
                    logger.warning(f"Removing symbols with >10% missing data: {removed_symbols}")
                    self.prices = self.prices[valid_symbols]
                    self.symbols = list(valid_symbols)
                
                # Forward/backward fill remaining gaps
                self.prices = self.prices.ffill(limit=5)
                self.prices = self.prices.bfill(limit=5)
            
            # Drop any remaining rows with missing values
            self.prices = self.prices.dropna()
            
            if self.prices.empty:
                raise ValueError("No valid data remaining after cleaning")
            
            logger.info(f"Data download completed with {len(self.symbols)} valid symbols")
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise

    def calculate_returns(self) -> None:
        """Calculate monthly returns from price data."""
        try:
            if self.prices is None:
                raise ValueError("No price data available")
            
            # Resample to monthly frequency and calculate returns
            monthly_prices = self.prices.resample('ME').last()
            self.returns = monthly_prices.pct_change()
            
            # Drop first row (NaN values)
            self.returns = self.returns.dropna()
            
            # Verify we have enough data
            if len(self.returns) < 12:  # At least one year of data
                raise ValueError("Insufficient data for analysis")
                
            logger.info(f"Calculated returns for {len(self.returns)} months")
            
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            raise

    def calculate_statistics(self) -> Dict[str, np.ndarray]:
        """Calculate portfolio statistics."""
        try:
            if self.returns is None:
                raise ValueError("No returns data available")
            
            # Calculate expected returns (historical means)
            expected_returns = self.returns.mean()
            
            # Calculate covariance matrix
            covariance_matrix = self.returns.cov()
            
            # Convert to numpy arrays
            self.statistics = {
                'expected_returns': expected_returns.values,
                'covariance_matrix': covariance_matrix.values,
                'symbols': self.symbols
            }
            
            # Verify matrix properties
            if not np.allclose(covariance_matrix, covariance_matrix.T):
                raise ValueError("Covariance matrix is not symmetric")
            
            logger.info("Portfolio statistics calculated successfully")
            logger.info(f"Statistics shapes - returns: {expected_returns.shape}, "
                       f"covariance: {covariance_matrix.shape}")
            
            return self.statistics
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            raise

    def save_data(self) -> None:
        """Save data to files."""
        try:
            processed_dir = self.data_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            
            if self.prices is not None:
                self.prices.to_csv(processed_dir / "prices.csv", index=True)
            if self.returns is not None:
                self.returns.to_csv(processed_dir / "returns.csv", index=True)
            if self.statistics is not None:
                for name, data in self.statistics.items():
                    if isinstance(data, np.ndarray):
                        pd.DataFrame(
                            data,
                            index=self.symbols,
                            columns=self.symbols if len(data.shape) > 1 else None
                        ).to_csv(processed_dir / f"{name}.csv")
            
            logger.info("Data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise


    def load_data(self) -> bool:
        """Load data from files."""
        try:
            processed_dir = self.data_dir / "processed"
            
            if not (processed_dir / "prices.csv").exists():
                return False
            
            # Load data preservando el tipo de índice
            self.prices = pd.read_csv(
                processed_dir / "prices.csv",
                index_col=0,
                parse_dates=True
            )
            self.prices.index = pd.DatetimeIndex(self.prices.index)
            
            self.returns = pd.read_csv(
                processed_dir / "returns.csv",
                index_col=0,
                parse_dates=True
            )
            self.returns.index = pd.DatetimeIndex(self.returns.index)
            
            # Load statistics
            expected_returns = pd.read_csv(
                processed_dir / "expected_returns.csv", 
                index_col=0
            )
            covariance_matrix = pd.read_csv(
                processed_dir / "covariance_matrix.csv", 
                index_col=0
            )
            
            # Update symbols from saved data
            self.symbols = expected_returns.index.tolist()
            
            self.statistics = {
                'expected_returns': expected_returns.values.flatten(),
                'covariance_matrix': covariance_matrix.values,
                'symbols': self.symbols
            }
            
            logger.info(f"Data loaded successfully for {len(self.symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def process_all(self) -> Dict[str, np.ndarray]:
        """Execute complete data processing pipeline."""
        if not self.load_data():
            self.download_data()
            self.calculate_returns()
            self.calculate_statistics()
            self.save_data()
        return self.statistics