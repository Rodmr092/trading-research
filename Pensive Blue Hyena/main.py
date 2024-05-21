# region imports
from AlgorithmImports import *
# endregion

class EjerciciosGPT(QCAlgorithm):

    
    def Initialize(self):
        # Inicializaciones y configuraciones aquí
        tickers = ["SPY", "BND"]
        self.equities = {}
        for ticker in tickers:
            self.equities[ticker] = self.AddEquity(ticker, Resolution.Daily).Symbol

    def OnData(self, data):
        # Número de equities con datos nuevos
        active_equities = 0

        # Verificar cuántas equities tienen datos nuevos
        for symbol in self.equities.values():
            if data.ContainsKey(symbol):
                active_equities += 1

        # Evitar división por cero si no hay equities activas
        if active_equities == 0:
            return

        # Calcular la proporción del valor total de la cartera a asignar a cada equity
        proportion_per_equity = 1.0 / active_equities

        # Asignar la proporción calculada a cada equity activa
        for symbol in self.equities.values():
            if data.ContainsKey(symbol):
                self.SetHoldings(symbol, proportion_per_equity)
