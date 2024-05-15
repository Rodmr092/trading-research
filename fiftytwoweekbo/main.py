# region imports
from AlgorithmImports import *
# endregion

class fiftytwoweekBO(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(100000)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        #inicializar indicador
        self.sma = self.SMA(self.spy, 30, Resolution.Daily)
        closing_prices = self.History(self.spy, 30, Resolution.Daily)["close"]
        for time, price in closing_prices.loc[self.spy].items():
            self.sma.Update(time, price)
        
        # Inicializar indicadores MIN y MAX
        self.minPrice = self.MIN(self.spy, 252, Resolution.Daily)
        self.maxPrice = self.MAX(self.spy, 252, Resolution.Daily)
        # Calentar los indicadores con datos históricos
        self.WarmUpIndicators(self.spy, 252, Resolution.Daily)

    def WarmUpIndicators(self, symbol, period, resolution):
        # Solicitar datos históricos
        history = self.History(symbol, period, resolution)
        for index, row in history.loc[symbol].iterrows():
            # Extraer el precio alto y bajo
            highPrice = row["high"]
            lowPrice = row["low"]
            
            # Actualizar los indicadores MIN y MAX
            self.minPrice.Update(index, lowPrice)
            self.maxPrice.Update(index, highPrice)

        
    def OnData(self, data: Slice):
        if not self.sma.IsReady:
            return

        #Guardar el precio del activo
        price = self.Securities[self.spy].Price
        # Obtener los valores actuales de los indicadores MIN y MAX
        low = self.minPrice.Current.Value
        high = self.maxPrice.Current.Value

        # Lógica de compra y venta, que esté sobre el high y esté encima del moving average:
        if price * 1.05 >= high and self.sma.Current.Value < price:     
            #si no está en la cartera
            if not self.Portfolio[self.spy].IsLong:
                self.SetHoldings(self.spy, 1)
        elif price * 0.95 <= low and self.sma.Current.Value > price:
            if not self.Portfolio[self.spy].IsShort:
                self.SetHoldings(self.spy, -1)
        else:
            self.Liquidate()

        self.Plot("Benchmark", "52w-High", high)
        self.Plot("Benchmark", "52w-Low", low)
        self.Plot("Benchmark", "SMA", self.sma.Current.Value)
