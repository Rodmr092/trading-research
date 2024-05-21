#region imports
from AlgorithmImports import *
#endregion
# 08/29/2023: -Adjusted insight expiry so all insights end at the same time each day
#             https://www.quantconnect.com/terminal/processCache?request=embedded_backtest_ea74a647c8c86a91ed350956de2dc6d4.html
# 
# 09/05/2023: -Replaced history request with consolidators + warm-up to avoid adjusting timezones of the data
#             https://www.quantconnect.com/terminal/processCache?request=embedded_backtest_2a472ff03a116c32320ff38bad217358.html
