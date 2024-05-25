#region imports
from AlgorithmImports import *
#endregion
from datetime import datetime, timedelta
import pandas as pd
from scipy.stats import skew

def calculate_expiration_date(year, month):
    first_day_of_month = datetime(year, month, 1)
    first_friday = first_day_of_month + timedelta(days=(4 - first_day_of_month.weekday() + 7) % 7)
    third_friday = first_friday + timedelta(weeks=2)
    return third_friday

expiration_dates = {
    'ES': [calculate_expiration_date(year, month) for year in range(2007, 2024) for month in [3, 6, 9, 12]],
    'MES': [calculate_expiration_date(year, month) for year in range(2019, 2024) for month in [3, 6, 9, 12]]
}

def calculate_roll_date(expiration_date):
    roll_date = expiration_date - timedelta(weeks=1)
    return roll_date
    
def cumulated_returns(symbol, previous_price, current_price, instruments):
    params = instruments[symbol]
    delta_price = current_price - previous_price
    return_value = delta_price * params['multiplier']
    return return_value

def calculate_return(symbol, price_series, instruments):
    params = instruments.get(symbol, {})
    multiplier = params.get('multiplier', 1)
    # Calculate daily returns in price points
    price_returns = price_series.diff().dropna()
    # Calculate daily returns in currency
    currency_returns = price_returns * multiplier
    # Create a DataFrame
    return_df = pd.DataFrame({
        'price_return': price_returns,
        'currency_return': currency_returns
    })
    return return_df

def calculate_percentage_return(symbol, price_series, instruments, num_contracts):
    # Extract the multiplier from the instruments dictionary
    params = instruments.get(symbol, {})
    multiplier = params.get('multiplier', 1)
    # Calculate changes in the price series
    price_changes = price_series.diff().fillna(0)  # Use fillna(0) to handle the NaN value of the first observation
    # Calculate returns in dollars for each time step
    returns_in_dollars = price_changes * multiplier * num_contracts
    # Calculate percentage returns based on the previous observation's price
    percentage_returns = (returns_in_dollars / (price_series.shift(1) * multiplier * num_contracts).fillna(1)) * 100
    # Return the series of percentage returns
    return percentage_returns

def calculate_and_merge_returns(symbol, price_series, instruments):
    """
    Inputs 
    symbol: str - a symbol to calculate returns for
    price_series: pd.Series - a price series to calculate returns from
    instruments: dict - a dictionary of instruments with metadata

    Returns
    merged_df: pd.DataFrame - a DataFrame with close, return, and percentage return columns
    """
    base_returns = calculate_return(symbol, price_series, instruments)
    percentage_returns = calculate_percentage_return(symbol, price_series, instruments, num_contracts=1)
    
    # Merge the DataFrames based on the index
    merged_df = base_returns.join(percentage_returns.rename('percentage_return'))
    # Include the original close price series
    merged_df['close'] = price_series
    
    return merged_df




def calculate_transaction_costs(params, history_df, start_date, end_date, num_contracts=1):
    # Convert to datetime if the inputs are tuples
    if isinstance(start_date, tuple):
        start_date = datetime(*start_date)
    if isinstance(end_date, tuple):
        end_date = datetime(*end_date)

    expiration_dates = params['expiration_dates']
    multiplier = params['multiplier']
    spread = params['spread']
    commission = params['commission']

    transaction_costs = 0
    for expiration_date in expiration_dates:
        roll_date = calculate_roll_date(expiration_date)
        if start_date <= roll_date <= end_date:
            transaction_costs += ((spread * multiplier) + commission) * 2 * num_contracts
    return transaction_costs

def calculate_statistics(df, trading_days_per_year):
    # Calculate the mean of the percentage returns
    mean_return = df['percentage_return'].mean()
    # Calculate the standard deviation of the percentage returns
    std_dev = df['percentage_return'].std()
    # Annualize the mean return and standard deviation
    annualized_mean = mean_return * trading_days_per_year
    annualized_std_dev = std_dev * np.sqrt(trading_days_per_year)
    # Calculate the daily Sharpe ratio (assuming risk-free rate is 0)
    daily_sharpe_ratio = mean_return / std_dev
    # Calculate the annualized Sharpe ratio
    annualized_sharpe_ratio = daily_sharpe_ratio * np.sqrt(trading_days_per_year)
    
    return mean_return, std_dev, annualized_mean, annualized_std_dev, daily_sharpe_ratio, annualized_sharpe_ratio

def calculate_fat_tail_ratios(df, mean_return):
    demeaned_returns = df['percentage_return'] - mean_return
    demeaned_returns = demeaned_returns.dropna()

    if len(demeaned_returns) > 0:
        p1 = np.percentile(demeaned_returns, 1)
        p30 = np.percentile(demeaned_returns, 30)
        p70 = np.percentile(demeaned_returns, 70)
        p99 = np.percentile(demeaned_returns, 99)

        lpr = p1 / p30 if p30 != 0 else np.nan
        upr = p99 / p70 if p70 != 0 else np.nan

        lower_fat_tail_ratio = lpr / 4.43 if not np.isnan(lpr) else np.nan
        higher_fat_tail_ratio = upr / 4.43 if not np.isnan(upr) else np.nan
    else:
        lower_fat_tail_ratio = np.nan
        higher_fat_tail_ratio = np.nan
    
    return lower_fat_tail_ratio, higher_fat_tail_ratio


def calculate_monthly_skew(df):
    df = df.reset_index(level='symbol', drop=True)
    monthly_returns = df['percentage_return'].resample('M').apply(lambda x: ((x + 1).prod() - 1) * 100)
    monthly_skew = skew(monthly_returns.dropna())
    
    return monthly_skew

def summarize_statistics(timeseries, trading_days_per_year):
    summary = {}

    for symbol, df in timeseries.items():
        mean_return, std_dev, annualized_mean, annualized_std_dev, daily_sharpe_ratio, annualized_sharpe_ratio = calculate_statistics(df, trading_days_per_year)
        lower_fat_tail_ratio, higher_fat_tail_ratio = calculate_fat_tail_ratios(df, mean_return)
        monthly_skew = calculate_monthly_skew(df)
        
        summary[symbol] = {
            'mean_return': mean_return,
            'std_dev': std_dev,
            'annualized_mean': annualized_mean,
            'annualized_std_dev': annualized_std_dev,
            'daily_sharpe_ratio': daily_sharpe_ratio,
            'annualized_sharpe_ratio': annualized_sharpe_ratio,
            'monthly_skew': monthly_skew,
            'lower_fat_tail_ratio': lower_fat_tail_ratio,
            'higher_fat_tail_ratio': higher_fat_tail_ratio
        }

    return summary

def calculate_contract_risk(summary, instruments, timeseries):
    """
    Calculate daily and annualized contract risk for each symbol and update the summary dictionary.
    
    Inputs:
    summary: dict - the dictionary of summary statistics
    instruments: dict - the dictionary of instruments with metadata
    timeseries: dict - the dictionary of time series data
    
    Returns:
    summary: dict - the updated summary dictionary with contract risks
    """
    for symbol in summary.keys():
        multiplier = instruments.get(symbol, {}).get('multiplier', 1)
        last_closing_price = timeseries[symbol]['close'].iloc[-1]
        std_dev = summary[symbol]['std_dev'] / 100  # Convert percentage to decimal
        annualized_std_dev = summary[symbol]['annualized_std_dev'] / 100  # Convert percentage to decimal

        daily_contract_risk = multiplier * last_closing_price * std_dev
        annualized_contract_risk = multiplier * last_closing_price * annualized_std_dev

        summary[symbol]['daily_contract_risk'] = daily_contract_risk
        summary[symbol]['annualized_contract_risk'] = annualized_contract_risk

    return summary



if __name__ == '__main__':
    print("This is a helper module. Import it into your main program.")
