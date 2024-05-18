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




if __name__ == '__main__':
    print("This is a helper module. Import it into your main program.")