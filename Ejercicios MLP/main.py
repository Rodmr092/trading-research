# region imports
from AlgorithmImports import *
# endregion

class EjerciciosMLP(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        future = self.AddFuture(Futures.Indices.SP500EMini)
        future.SetFilter(timedelta(0), timedelta(182))
        future.Resolution = Resolution.Tick
       
        self.symbols = []
        self.current_contract = None
        self.next_contract = None
        self.roll_dates = self.GenerateRollDates()
        self.holdings = {}
        self.value_of_etf = 1.0

    def GenerateRollDates(self):
        roll_dates = {}
        for year in range(self.StartDate.year, self.EndDate.year + 1):
            for month in range(1, 13):
                third_friday = self.ThirdFriday(year, month)
                roll_dates[third_friday] = True
        return roll_dates
    
    def ThirdFriday(self, year, month):
        # Ensure correct calculation for the third Friday
        third_friday = datetime(year, month, 1)
        # How many days to add to reach the first Friday of the month
        days_to_add = (4 - third_friday.weekday() + 7) % 7
        third_friday += timedelta(days=days_to_add)
        # Adding two weeks to reach the third Friday
        third_friday += timedelta(weeks=2)
        return third_friday



    def OnData(self, slice):
        if self.current_contract is not None:
            ticks = slice.Ticks[self.current_contract.Symbol]
            for tick in ticks:
                # Here you could update ETF value based on the tick data
                # This is a simplified example; you'll likely need a more complex logic
                pass  # Placeholder for processing logic

        # Handle roll logic separately from tick processing
        for chain in slice.FutureChains:
            for contract in chain.Value:
                if self.current_contract is None or contract.Symbol == self.current_contract.Symbol:
                    if contract.Expiry.date() in self.roll_dates:
                        self.HandleRoll(contract)
                    break  # Assume we only handle one contract at a time

    def HandleRoll(self, contract):
        # If we're on the roll date and we have a current contract, liquidate it
        if self.current_contract and self.Time.date() in self.roll_dates:
            self.Liquidate(self.current_contract.Symbol)
        self.current_contract = contract
        self.holdings = {contract.Symbol: self.value_of_etf / contract.Price}

    def CalculateHoldings(self, contract):
        # If we have current holdings, update the value of the holdings based on the current price
        if self.current_contract and self.current_contract.Symbol in self.holdings:
            self.holdings[self.current_contract.Symbol] = self.value_of_etf / contract.Price

    def UpdateValueOfETF(self, contract):
        # If we have current holdings, update the value of the ETF
        if self.current_contract and self.current_contract.Symbol in self.holdings:
            self.value_of_etf = self.holdings[self.current_contract.Symbol] * contract.Price

    def OnEndOfDay(self):
        self.Log(f"ETF Value: {self.value_of_etf}")


