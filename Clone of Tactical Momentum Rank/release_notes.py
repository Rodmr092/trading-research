#region imports
from AlgorithmImports import *
#endregion
# 05/19/2023: -Added a warm-up period to restore the algorithm state between deployments.
#             -Added OnWarmupFinished to liquidate existing holdings that aren't backed by active insights.
#             -Removed flat insights because https://github.com/QuantConnect/Lean/pull/7251 made them unnecessary.
#             https://www.quantconnect.com/terminal/processCache?request=embedded_backtest_a34c371a3b4818e5157cd76b876ecae0.html
#
# 07/13/2023: -Replaced the SymbolData class by with custom Security properties
#             -Fixed warm-up logic to liquidate undesired portfolio holdings on re-deployment
#             -Set the MinimumOrderMarginPortfolioPercentage to 0
#             https://www.quantconnect.com/terminal/processCache?request=embedded_backtest_82183246d97159739b71348a0a09c64a.html 
