# region imports
from AlgorithmImports import *

#from futures import future_datas
from universe import AdvancedFuturesUniverseSelectionModel
from alpha import FastTrendFollowingLongAndShortWithTrendStrenthAlphaModel
from portfolio import BufferedPortfolioConstructionModel
# endregion

class FuturesFastTrendFollowingLongAndShortWithTrendStrenthAlgorithm(QCAlgorithm):

    undesired_symbols_from_previous_deployment = []
    checked_symbols_from_previous_deployment = False
    futures = []
    
    def Initialize(self):
        self.SetStartDate(2020, 7, 1)
        self.SetEndDate(2023, 7, 1)
        self.SetCash(1_000_000)

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetSecurityInitializer(BrokerageModelSecurityInitializer(self.BrokerageModel, FuncSecuritySeeder(self.GetLastKnownPrices)))        
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0

        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.BackwardsPanamaCanal
        self.UniverseSettings.DataMappingMode = DataMappingMode.OpenInterest
        self.AddUniverseSelection(AdvancedFuturesUniverseSelectionModel())

        
        slow_ema_span = 2 ** self.GetParameter("slow_ema_span_exponent", 6) # Should be >= 5. "It's convenient to stick to a series of parameter values that are powers of two" (p.131)
        blend_years = self.GetParameter("blend_years", 3)                   # Number of years to use when blending sigma estimates
        self.AddAlpha(FastTrendFollowingLongAndShortWithTrendStrenthAlphaModel(
            self,
            slow_ema_span, 
            self.GetParameter("abs_forecast_cap", 20),           # Hardcoded on p.173
            self.GetParameter("sigma_span", 32),                 # Hardcoded to 32 on p.604
            self.GetParameter("target_risk", 0.2),               # Recommend value is 0.2 on p.75
            blend_years
        ))

        self.Settings.RebalancePortfolioOnSecurityChanges = False
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.total_count = 0
        self.day = -1
        self.SetPortfolioConstruction(BufferedPortfolioConstructionModel(
            self.rebalance_func,
            self.GetParameter("buffer_scaler", 0.1)              # Hardcoded on p.167 & p.173
        ))

        self.AddRiskManagement(NullRiskManagementModel())

        self.SetExecution(ImmediateExecutionModel())

        self.SetWarmUp(timedelta(365*blend_years + slow_ema_span + 7))

    def rebalance_func(self, time):
        if (self.total_count != self.Insights.TotalCount or self.day != self.Time.day) and not self.IsWarmingUp and self.CurrentSlice.QuoteBars.Count > 0:
            self.total_count = self.Insights.TotalCount
            self.day = self.Time.day
            return time
        return None

    def OnData(self, data):
        # Exit positions that aren't backed by existing insights.
        # If you don't want this behavior, delete this method definition.
        if not self.IsWarmingUp and not self.checked_symbols_from_previous_deployment:
            for security_holding in self.Portfolio.Values:
                if not security_holding.Invested:
                    continue
                symbol = security_holding.Symbol
                if not self.Insights.HasActiveInsights(symbol, self.UtcTime):
                    self.undesired_symbols_from_previous_deployment.append(symbol)
            self.checked_symbols_from_previous_deployment = True
        
        for symbol in self.undesired_symbols_from_previous_deployment[:]:
            if self.IsMarketOpen(symbol):
                self.Liquidate(symbol, tag="Holding from previous deployment that's no longer desired")
                self.undesired_symbols_from_previous_deployment.remove(symbol)

