#region imports
from AlgorithmImports import *
#endregion
# 08/29/2023: -Adjusted insight expiry so all insights end at the same time each day
#             https://www.quantconnect.com/terminal/processCache?request=embedded_backtest_e1c8af207b1a4da945a4696f7db3ef9a.html
#
# 08/31/2023: -Adjusted universe filter to ensure the Mapped contract is always in the universe
#             -Updated the Alpha model to rely on warm-up rather than history requests
#             -Reduced the `blend_years` parameter to 3 to avoid any data issues from far in the past
#             https://www.quantconnect.com/terminal/processCache?request=embedded_backtest_ecb85ecf7a6ea332088f4b369017fa09.html 

