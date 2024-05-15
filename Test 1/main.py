# region imports
from AlgorithmImports import *
# endregion

class Test1(QCAlgorithm):

    def Initialize(self):
        # Locally Lean installs free sample data, to download more data please visit https://www.quantconnect.com/docs/v2/lean-cli/datasets/downloading-data
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity("SPY", Resolution.Minute)
        self.AddEquity("BAC", Resolution.Minute)
        self.AddEquity("IBM", Resolution.Minute)

    def OnData(self, data: Slice):
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 0.33)
            self.SetHoldings("BAC", 0.33)
            self.SetHoldings("IBM", 0.33)
