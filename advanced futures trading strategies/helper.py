#region imports
from AlgorithmImports import *
#endregion
from datetime import datetime, timedelta
import pandas as pd
from scipy.stats import skew


# Function to compute FX change to MXN
def compute_fx_change(returns, timeseries):
    fx_rate = timeseries['close'].iloc[-1]  # Extract the last closing price of USDMXN
    converted_returns = returns * fx_rate
    return converted_returns
    
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
    # Handle NaN and infinite values
    percentage_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    percentage_returns.dropna(inplace=True)
    # Debug: Check if percentage_returns is empty
    if percentage_returns.empty:
        print(f"Warning: Calculated percentage_returns is empty for symbol {symbol}")
    # Return the series of percentage returns
    return percentage_returns




def calculate_and_merge_returns(symbol, price_series, volume_series, instruments):
    """
    Inputs 
    symbol: str - a symbol to calculate returns for
    price_series: pd.Series - a price series to calculate returns from
    volume_series: pd.Series - a volume series to include in the final DataFrame
    instruments: dict - a dictionary of instruments with metadata

    Returns
    merged_df: pd.DataFrame - a DataFrame with close, volume, return, and percentage return columns
    """
    # Calculate base returns
    base_returns = calculate_return(symbol, price_series, instruments)
    # Calculate percentage returns
    percentage_returns = calculate_percentage_return(symbol, price_series, instruments, num_contracts=1)
    # Check if percentage_returns is not empty
    if percentage_returns.empty:
        print(f"Warning: percentage_returns is empty for symbol {symbol}")
    # Merge the DataFrames based on the index
    merged_df = base_returns.join(percentage_returns.rename('percentage_return'), how='left')
    # Include the original close price series and volume series
    merged_df['close'] = price_series
    merged_df['volume'] = volume_series
    # Handle NaN values after merging
    merged_df.dropna(subset=['percentage_return'], inplace=True)
    # Debug: Check if 'percentage_return' column exists after merging
    if 'percentage_return' not in merged_df.columns:
        print(f"Error: 'percentage_return' column missing for symbol {symbol} after merging")
    
    return merged_df



def calculate_rolling_costs(instrument_details, timeseries):
    if 'rolling_months' not in instrument_details:
        return 0  # No rolling months, hence no rolling costs
    
    rolling_costs = 0
    rolling_months = instrument_details['rolling_months']
    years = range(timeseries.index[0][1].year, timeseries.index[-1][1].year + 1)
    
    for year in years:
        for month in rolling_months:
            # Find the first available trading day in the rolling month
            roll_date = pd.Timestamp(year=year, month=month, day=1)
            found = False
            while roll_date.month == month:
                if roll_date.date() in timeseries.index.get_level_values(1).date:
                    cost_per_roll = instrument_details['commission'] + (instrument_details['spread'] / 2) * instrument_details['multiplier']
                    rolling_costs += cost_per_roll * 2  # Buy and Sell
                    found = True
                    break
                roll_date += pd.Timedelta(days=1)
    
    return rolling_costs



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


def simulate_buy_and_hold(timeseries, instrument_details, fx_timeseries, num_contracts=1):
    """
    Function to simulate buy and hold strategy for a futures contract
    
    inputs:
    timeseries: pandas DataFrame containing the price data
    instrument_details: dictionary containing the instrument details
    fx_timeseries: pandas DataFrame containing the forex data for currency conversion
    num_contracts: number of contracts to trade

    outputs:
    net_return_base: net return in base currency
    net_return_mxn: net return in MXN
    """
    first_price = timeseries['close'].iloc[0]
    last_price = timeseries['close'].iloc[-1]

    # Calculate rolling costs
    rolling_costs = calculate_rolling_costs(instrument_details, timeseries)
    
    # Calculate net return in base currency
    net_return_base = (last_price - first_price) * instrument_details['multiplier'] * num_contracts - rolling_costs
    
    # Convert net return to MXN
    net_return_mxn = compute_fx_change(net_return_base, fx_timeseries)
    
    return net_return_base, net_return_mxn

def calculate_short_term_ewma(data):
    """
    Calculate the short-term EWMA (32 periods) for the percentage returns and its standard deviation.
    Returns two series: ewma_32 and ewma_32_std.
    """
    ewma_32 = data['percentage_return'].ewm(span=32, adjust=False).mean()
    ewma_32_std = ewma_32.ewm(span=32, adjust=False).std()
    return ewma_32, ewma_32_std

def calculate_long_term_ewma(data):
    """
    Calculate the long-term EWMA (2560 periods) for the percentage returns and its standard deviation.
    Returns two series: ewma_2560 and ewma_2560_std.
    """
    ewma_2560 = data['percentage_return'].ewm(span=2560, adjust=False).mean()
    ewma_2560_std = ewma_2560.ewm(span=2560, adjust=False).std()
    return ewma_2560, ewma_2560_std


def calculate_combined_std(timeseries):
    """
    Calculate the combined standard deviation of returns using 0.3*long_term_ewma + 0.7*short_term_ewma.
    Returns a dictionary with symbols as keys and combined standard deviation as values.
    """
    combined_std = {}
    for symbol, data in timeseries.items():
        ewma_32, ewma_32_std = calculate_short_term_ewma(data)
        ewma_2560, ewma_2560_std = calculate_long_term_ewma(data)
        combined_std[symbol] = 0.3 * ewma_2560_std + 0.7 * ewma_32_std
    return combined_std

if __name__ == '__main__':
    print("This is a helper module. Import it into your main program.")
