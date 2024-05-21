#region imports
from AlgorithmImports import *
#endregion

class MomentumQuantilesAlphaModel(AlphaModel):

    securities = []
    day = -1

    def __init__(self, quantiles, lookback_months):
        self.quantiles = quantiles
        self.lookback_months = lookback_months

    def Update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        # Reset indicators when corporate actions occur
        for symbol in set(data.Splits.keys() + data.Dividends.keys()):
            security = algorithm.Securities[symbol]
            if security in self.securities:
                security.indicator.Reset()
                algorithm.SubscriptionManager.RemoveConsolidator(security.Symbol, security.consolidator)
                self.register_indicator(algorithm, security)

                history = algorithm.History[TradeBar](security.Symbol, (security.indicator.WarmUpPeriod+1) * 30, Resolution.Daily, dataNormalizationMode=DataNormalizationMode.ScaledRaw)
                for bar in history:
                    security.consolidator.Update(bar)
        
        # Only emit insights when there is quote data, not when a corporate action occurs (at midnight)
        if data.QuoteBars.Count == 0:
            return []
        
        # Only emit insights once per day
        if self.day == algorithm.Time.day:
            return []
        self.day = algorithm.Time.day

        # Get the momentum of each asset in the universe
        momentum_by_symbol = {security.Symbol : security.indicator.Current.Value 
            for security in self.securities if security.Symbol in data.QuoteBars and security.indicator.IsReady}
                
        # Determine how many assets to hold in the portfolio
        quantile_size = int(len(momentum_by_symbol)/self.quantiles)
        if quantile_size == 0:
            return []

        # Create insights to long the assets in the universe with the greatest momentum
        weight = 1 / quantile_size
        expiry = self.securities[0].Exchange.Hours.GetNextMarketOpen(algorithm.Time, False) - timedelta(seconds=1)
        insights = []
        for symbol, _ in sorted(momentum_by_symbol.items(), key=lambda x: x[1], reverse=True)[:quantile_size]:
            insights.append(Insight.Price(symbol, expiry, InsightDirection.Up, weight=weight))

        return insights

    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        # Create and register indicator for each security in the universe
        security_by_symbol = {}
        for security in changes.AddedSecurities:
            security_by_symbol[security.Symbol] = security
            
            # Create an indicator that automatically updates each month
            security.indicator = MomentumPercent(self.lookback_months)
            self.register_indicator(algorithm, security)

            self.securities.append(security)
        
        # Warm up the indicators of newly-added stocks
        if security_by_symbol:
            history = algorithm.History[TradeBar](list(security_by_symbol.keys()), (self.lookback_months+1) * 30, Resolution.Daily, dataNormalizationMode=DataNormalizationMode.ScaledRaw)
            for trade_bars in history:
                for bar in trade_bars.Values:
                    security_by_symbol[bar.Symbol].consolidator.Update(bar)

        # Stop updating consolidator when the security is removed from the universe
        for security in changes.RemovedSecurities:
            if security in self.securities:
                algorithm.SubscriptionManager.RemoveConsolidator(security.Symbol, security.consolidator)
                self.securities.remove(security)


    def register_indicator(self, algorithm, security):
        # Update the indicator with monthly bars
        security.consolidator = TradeBarConsolidator(Calendar.Monthly)
        algorithm.SubscriptionManager.AddConsolidator(security.Symbol, security.consolidator)
        algorithm.RegisterIndicator(security.Symbol, security.indicator, security.consolidator)

        
