#region imports
from AlgorithmImports import *
#endregion
from datetime import datetime, timedelta
import pandas as pd

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
    params = instruments[symbol]
    multiplier = params['multiplier']
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
    # Extraer el multiplicador del diccionario de instrumentos
    params = instruments.get(symbol, {})
    multiplier = params.get('multiplier', 1)

    # Calcular los cambios en la serie de precios
    price_changes = price_series.diff().fillna(0)  # Usar fillna(0) para manejar el valor NaN de la primera observación

    # Calcular retornos en dólares para cada paso de tiempo
    returns_in_dollars = price_changes * multiplier * num_contracts

    # Calcular los retornos porcentuales basados en el precio de la observación anterior
    percentage_returns = (returns_in_dollars / (price_series.shift(1) * multiplier * num_contracts).fillna(1)) * 100

    # Retornar la serie de retornos porcentuales
    return percentage_returns



def calculate_and_merge_returns(symbol, price_series, instruments):
    """
    Inputs 
    symbol: str - a symbol to calculate returns for
    price_series: pd.Series - a price series to calculate returns from
    instruments: dict - a dictionary of instruments with metadata

    Returns
    merged_df: pd.DataFrame - a DataFrame with both return and percentage return columns
    """
    returns_df = calculate_return(symbol, price_series, instruments)
    percentage_returns = calculate_percentage_return(symbol, price_series, instruments, num_contracts=1)
    
    # Unir los DataFrames en base al índice
    merged_df = returns_df.join(percentage_returns.rename('percentage_return'))
    # Clean up the memory
    del returns_df
    del percentage_returns
    return merged_df

def calculate_return_in_usd(symbol, initial_price, final_price, instruments, num_contracts=1):
    params = instruments[symbol]
    delta_price = final_price - initial_price
    return_in_usd = delta_price * params['multiplier'] * num_contracts
    return return_in_usd

def calculate_return_in_mxn(return_in_usd, fx_rates_df):
    average_fx_rate = fx_rates_df['close'].mean()
    return_in_mxn = return_in_usd * average_fx_rate
    return return_in_mxn


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



if __name__ == '__main__':
    print("This is a helper module. Import it into your main program.")
