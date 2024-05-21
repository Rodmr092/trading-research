import unittest
import numpy as np
import pandas as pd

# Function to determine the trend based on the N-day window
def determine_trend(data, N):
    highest_high = data['high'].rolling(window=N).max()
    lowest_low = data['low'].rolling(window=N).min()
    trend = np.where(data['high'] > highest_high, 'uptrend', 
                     np.where(data['low'] < lowest_low, 'downtrend', 'no_trend'))
    return trend

# Function to generate signals based on the n-day window
def generate_signals(data, trend, n):
    highest_high_n = data['high'].rolling(window=n).max()
    lowest_low_n = data['low'].rolling(window=n).min()
    
    long_signals = (data['high'] > highest_high_n) & (trend == 'uptrend')
    short_signals = (data['low'] < lowest_low_n) & (trend == 'downtrend')

    return long_signals, short_signals



class TestNDayBreakoutStrategy(unittest.TestCase):

    def test_determine_trend(self):
        data = pd.DataFrame({
            'high': [120, 125, 130, 128, 135, 132],
            'low': [115, 117, 118, 123, 125, 126]
        })
        expected_trends = ['no_trend', 'no_trend', 'uptrend', 'no_trend', 'uptrend', 'no_trend']
        actual_trends = determine_trend(data, 3)
        self.assertEqual(actual_trends.tolist(), expected_trends)

    def test_generate_signals(self):
        data = pd.DataFrame({
            'high': [120, 125, 130, 128, 135, 132],
            'low': [115, 117, 118, 123, 125, 126]
        })
        trend = ['no_trend', 'no_trend', 'uptrend', 'no_trend', 'uptrend', 'downtrend']
        expected_long_signals = [False, False, True, False, True, False]
        expected_short_signals = [False, False, False, False, False, True]
        long_signals, short_signals = generate_signals(data, trend, 2)
        self.assertEqual(long_signals.tolist(), expected_long_signals)
        self.assertEqual(short_signals.tolist(), expected_short_signals)

# This line allows you to run the tests when this script is executed
if __name__ == '__main__':
    unittest.main()
