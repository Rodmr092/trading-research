# region imports
from AlgorithmImports import *
# endregion

class CreatingSingleContractDollarBars(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        #Agregar SP500 al algoritmo
        self.es = self.AddFuture(Futures.Indices.SP500EMini, Resolution.Minute)
        self.es.Symbol = "ES"

        #Definir cuántos datos históricos obtener:
        self.history_window = TimeSpan.FromDays(3 * 252)  # Historical datapoints (in minutes)
        self.Plot("Test Chart", "Test Value", 10)  # This generates a single point


    def OnData(self, data: slice):
        if not self.Portfolio.Invested:
            if self.IsMarketOpen(self.es.Symbol): #Check if the market is open
                if data.Bars.ContainsKey(self.es.Symbol):
                    bar = data.Bars[self.es.Symbol]
                    
                    #Fetch Historical data
                    history_df = self.History(self.es.Symbol, self.history_window, Resolution.Minute)
                    #Extract close prices from the dataframe

                    close_price = history_df['close'].tolist()

                    print(f"Type of close_price: {type(close_price)}")  
                    print(f"Length of close_price: {len(close_price)}") 

                    if close_price:    
                        for i, price in  enumerate(close_price): 
                            print(f"Index: {i} , Price: {price}, Type: {type(price)}") 

                            try:
                                self.Plot("Price History", "Price", price)
                            except Exception as e:
                                print(f"Exception at index {i}: {e}")  