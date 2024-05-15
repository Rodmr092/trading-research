#region imports
from AlgorithmImports import *
from Portfolio.EqualWeightingPortfolioConstructionModel import EqualWeightingPortfolioConstructionModel
#endregion

class InverseVolatilityPortfolioConstructionModel(EqualWeightingPortfolioConstructionModel):
    def __init__(self, rebalance = Expiry.EndOfWeek, lookback = 30):
        '''Initialize a new instance of EqualWeightingPortfolioConstructionModel
        Args:
            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.
                              If None will be ignored.
                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.
                              The function returns null if unknown, in which case the function will be called again in the
                              next loop. Returning current time will trigger rebalance.
            lookback: The lookback period of historical return to calculate the volatility'''
        super().__init__(rebalance, PortfolioBias.Long)
        self.symbolData = {}
        self.lookback = lookback

    def DetermineTargetPercent(self, activeInsights):
        '''Will determine the target percent for each insight
        Args:
            activeInsights: The active insights to generate a target for'''
        result = {}

        active_symbols = [insight.Symbol for insight in activeInsights]
        active_data = {symbol: data for symbol, data in self.symbolData.items() 
            if data.IsReady and symbol in active_symbols}       # make sure data in used are ready

        # Sum the inverse STD of ROC for normalization later
        std_sum = sum([1 / data.Value for data in active_data.values()])

        if std_sum == 0:
            return {insight: 0 for insight in activeInsights}
        
        for insight in activeInsights:
            if insight.Symbol in active_data:
                data = active_data[insight.Symbol]
                # Sizing by inverse volatility, then divide by contract multiplier to avoid unwanted leveraging
                result[insight] = 1 / data.Value / std_sum / data.multiplier
            else:
                result[insight] = 0
        
        return result

    def OnSecuritiesChanged(self, algorithm, changes):
        super().OnSecuritiesChanged(algorithm, changes)
        for removed in changes.RemovedSecurities:
            data = self.symbolData.pop(removed.Symbol, None)
            # Free up resources
            if data:
                data.Dispose()

        for added in changes.AddedSecurities:
            symbol = added.Symbol
            if symbol not in self.symbolData:
                self.symbolData[symbol] = SymbolData(algorithm, added, self.lookback)

class SymbolData:
    '''An object to hold the daily return and volatility data for each security'''
    def __init__(self, algorithm, security, period):
        self.algorithm = algorithm
        self.Symbol = security.Symbol
        self.multiplier = security.SymbolProperties.ContractMultiplier

        self.ROC = RateOfChange(1)
        self.Volatility = IndicatorExtensions.Of(StandardDeviation(period), self.ROC)

        self.consolidator = TradeBarConsolidator(timedelta(1))
        self.consolidator.DataConsolidated += self.OnDataUpdate
        algorithm.SubscriptionManager.AddConsolidator(self.Symbol, self.consolidator)

        # Warm up with historical data
        history = algorithm.History[TradeBar](self.Symbol, period+1, Resolution.Daily)
        for bar in history:
            self.ROC.Update(bar.EndTime, bar.Close)

    def OnDataUpdate(self, sender, bar):
        self.ROC.Update(bar.EndTime, bar.Close)

    def Dispose(self):
        '''Free up memory and speed up update cycle'''
        self.consolidator.DataConsolidated -= self.OnDataUpdate
        self.algorithm.SubscriptionManager.RemoveConsolidator(self.Symbol, self.consolidator)
        self.ROC.Reset()
        self.Volatility.Reset()

    @property
    def IsReady(self):
        return self.Volatility.IsReady

    @property
    def Value(self):
        return self.Volatility.Current.Value
