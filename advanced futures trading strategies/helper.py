#region imports
from AlgorithmImports import *
#endregion


# Your New Python File
from datetime import datetime, timedelta

def calculate_expiration_date(year, month):
    """
    Calculate the third Friday of a given month and year. Returns datetime.
    """
    first_day_of_month = datetime(year, month, 1)
    first_friday = first_day_of_month + timedelta(days=(4 - first_day_of_month.weekday() + 7) % 7)
    third_friday = first_friday + timedelta(weeks=2)
    return third_friday

# Create a dictionary with the expiration dates for ES and MES
expiration_dates = {
    'ES': [calculate_expiration_date(year, month) for year in range(2007, 2024) for month in [3, 6, 9, 12]],
    'MES': [calculate_expiration_date(year, month) for year in range(2019, 2024) for month in [3, 6, 9, 12]]
}

def calculate_return(symbol, previous_price, current_price, instruments):
    """
    Compute returns. Takes symbol, beginning price, end price, and instruments dictionary. Returns float.
    """
    params = instruments[symbol]
    delta_price = current_price - previous_price
    return_value = delta_price * params['multiplier']
    return return_value

def calculate_roll_date(expiration_date):
    """
    Calculate the roll date (one week before expiration). Returns datetime.
    """
    roll_date = expiration_date - timedelta(weeks=1)
    return roll_date

def calculate_return_in_usd(symbol, initial_price, final_price, instruments):
    """
    Helper function to calculate returns in USD
    """
    params = instruments[symbol]
    delta_price = final_price - initial_price
    return_in_usd = delta_price * params['multiplier']
    return return_in_usd

def calculate_return_in_mxn(return_in_usd, fx_rates_df):
    """
    Helper function to calculate returns in MXN
    """
    average_fx_rate = fx_rates_df['close'].mean()
    return_in_mxn = return_in_usd * average_fx_rate
    return return_in_mxn

def calculate_transaction_costs(params, history_df, start_date, end_date):
    """
    Function to calculate transaction costs
    """
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




if __name__ == '__main__':
    print("This is a helper module. Import it into your main program.")