#region imports
from AlgorithmImports import *
#endregion
import unittest
import pandas as pd
from helper import calculate_percentage_return
# Your New Python File



class TestCalculatePercentageReturn(unittest.TestCase):
    def setUp(self):
        # Set up the test data and instruments dictionary
        self.instruments = {
            'ES': {
                'multiplier': 50,
                'tick_value': 0.25,
                'minimum_fluctuation': 12.50,
                'spread': 0.25,
                'commission': 2.50,
                'expiration_dates': []
            },
            'MES': {
                'multiplier': 5,
                'tick_value': 0.25,
                'minimum_fluctuation': 1.25,
                'spread': 0.25,
                'commission': 2.50,
                'expiration_dates': []
            }
        }
        self.initial_cash = 100000
        self.price_series_es = pd.Series([3000, 3100, 3200, 3300, 3400])
        self.price_series_mes = pd.Series([1500, 1550, 1600, 1650, 1700])
        self.num_contracts_es = 10
        self.num_contracts_mes = 20

    def test_calculate_percentage_return_es(self):
        expected_returns = ((self.price_series_es - self.price_series_es.iloc[0]) * 
                            self.instruments['ES']['multiplier'] * 
                            self.num_contracts_es / self.initial_cash) * 100
        actual_returns = calculate_percentage_return('ES', self.price_series_es, self.initial_cash, self.instruments, self.num_contracts_es)
        pd.testing.assert_series_equal(expected_returns, actual_returns)

    def test_calculate_percentage_return_mes(self):
        expected_returns = ((self.price_series_mes - self.price_series_mes.iloc[0]) * 
                            self.instruments['MES']['multiplier'] * 
                            self.num_contracts_mes / self.initial_cash) * 100
        actual_returns = calculate_percentage_return('MES', self.price_series_mes, self.initial_cash, self.instruments, self.num_contracts_mes)
        pd.testing.assert_series_equal(expected_returns, actual_returns)
    
    