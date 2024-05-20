#region imports
from AlgorithmImports import *
#endregion


# Your New Python File
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

def calculate_return(symbol, previous_price, current_price, instruments):
    params = instruments[symbol]
    delta_price = current_price - previous_price
    return_value = delta_price * params['multiplier']
    return return_value

def calculate_roll_date(expiration_date):
    roll_date = expiration_date - timedelta(weeks=1)
    return roll_date

def calculate_return_in_usd(symbol, initial_price, final_price, instruments):
    params = instruments[symbol]
    delta_price = final_price - initial_price
    return_in_usd = delta_price * params['multiplier']
    return return_in_usd

def calculate_return_in_mxn(return_in_usd, fx_rates_df):
    average_fx_rate = fx_rates_df['close'].mean()
    return_in_mxn = return_in_usd * average_fx_rate
    return return_in_mxn

def calculate_transaction_costs(params, history_df, start_date, end_date):
    # Convert to datetime if the inputs are tuples
    if isinstance(start_date, tuple):
        start_date = datetime(*start_date)
    if isinstance(end_date, tuple):
        end_date = datetime(*end_date)

    # Debugging print statements
    print(f"Start date type: {type(start_date)}")
    print(f"End date type: {type(end_date)}")

    expiration_dates = params['expiration_dates']
    multiplier = params['multiplier']
    spread = params['spread']
    commission = params['commission']

    transaction_costs = 0
    for expiration_date in expiration_dates:
        roll_date = calculate_roll_date(expiration_date)
        if start_date <= roll_date <= end_date:
            transaction_costs += ((spread * multiplier) + commission) * 2
    return transaction_costs



def calculate_percentage_returns(history_df, symbol, instruments, fx_rates_df, initial_cash):
    params = instruments[symbol]
    returns = []

    # Ensure history_df index is datetime
    history_df.index = pd.to_datetime(history_df.index)

    for i in range(1, len(history_df)):
        initial_price = history_df['askclose'].iloc[i-1]  # Use 'askclose' for prices based on your data structure
        final_price = history_df['askclose'].iloc[i]

        return_in_usd = calculate_return_in_usd(symbol, initial_price, final_price, instruments)
        return_in_mxn = calculate_return_in_mxn(return_in_usd, fx_rates_df)

        # Ensure that the dates are datetime objects
        start_date = pd.to_datetime(history_df.index[0])
        end_date = pd.to_datetime(history_df.index[i])

        if i == 1:
            initial_cash -= calculate_transaction_costs(params, history_df, start_date, end_date)

        percentage_return = return_in_mxn / initial_cash
        returns.append(percentage_return)

        initial_cash += return_in_mxn
        initial_cash -= calculate_transaction_costs(params, history_df, start_date, end_date)

    return pd.Series(returns, index=history_df.index[1:])


    
if __name__ == '__main__':
    print("This is a helper module. Import it into your main program.")
