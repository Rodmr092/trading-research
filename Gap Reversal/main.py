# region imports
from AlgorithmImports import *
# endregion

class GapReversal(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(100000)
        self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol
        #Inicializa Rolling window
        self.rollingWindow = RollingWindow[TradeBar](2)
        #Inicializa Consolidador
        self.Consolidate(self.symbol, Resolution.Daily, self.CustomBarHandler)
        #Para el cierre de posiciones:
        self.Schedule.On(self.DateRules.EveryDay(self.symbol), self.TimeRules.BeforeMarketClose(self.symbol, 15), self.ExitPositions)

    def OnData(self, data):
        # Revisar si está listo el rolling window
        if not self.rollingWindow.IsReady:
            return

        if not (self.Time.hour == 9 and self.Time.minute == 31):
            return

        # Check if `self.symbol` is present in `data` before accessing its attributes
        if self.symbol in data:
            # Verifica si el objeto tiene el atributo 'Open'
            tradeBar = data[self.symbol]
            if hasattr(tradeBar, 'Open'):
                # Trading logic
                if tradeBar.Open >= 1.01 * self.rollingWindow[0].Close:
                    self.SetHoldings(self.symbol, -1)
                elif tradeBar.Open <= 0.99 * self.rollingWindow[0].Close:
                    self.SetHoldings(self.symbol, 1)
            else:
                self.Debug(f"{self.Time} - Open data for {self.symbol.Value} not available.")
        else:
            self.Debug(f"{self.Time} - Data for {self.symbol.Value} not available.")

    #Event handler para el consolidador
    def CustomBarHandler (self, bar):
        self.rollingWindow.Add(bar)

    def ExitPositions (self):
        self.Liquidate(self.symbol)

