from AlgorithmImports import *

class FocusedTanBarracuda(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2000, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(100000)

        spy = self.AddEquity("SPY", Resolution.Daily)
        self.spy = spy.Symbol

        self.SetBenchmark(self.spy)  # Corregido para usar el símbolo directamente
        # self.SetBrokerageModel

        self.entryPrice = 0
        self.period = timedelta(31)
        self.nextEntryTime = self.Time

    def OnData(self, data: Slice):
        if self.spy not in data or data[self.spy] is None:  # Revisa si existe el data y no es None
            return

        bar = data[self.spy]
        if bar is not None and hasattr(bar, 'Close'):
            price = bar.Close

            if not self.Portfolio.Invested:
                if self.nextEntryTime <= self.Time:
                    self.SetHoldings(self.spy, 1)
                    self.Log("Buy SPY at " + str(price))
                    self.entryPrice = price

                elif self.entryPrice != 0 and (self.entryPrice * 1.1 < price or self.entryPrice * 0.9 > price):
                    self.Liquidate(self.spy)
                    self.Log("Sell SPY at " + str(price))
                    self.nextEntryTime = self.Time + self.period

