# region imports
from AlgorithmImports import *
# endregion

class JumpingYellowGreenPenguin(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2022, 8, 22)
        self.SetCash(100000)
        self.AddEquity("SPY", Resolution.Minute)
        self.AddEquity("BND", Resolution.Minute)
        self.AddEquity("AAPL", Resolution.Minute)

    def OnData(self, data: Slice):
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 0.33)
            self.SetHoldings("BND", 0.33)
            self.SetHoldings("AAPL", 0.33)
