# region imports
from AlgorithmImports import *

from universe import QQQConstituentsUniverseSelectionModel
from alpha import MomentumQuantilesAlphaModel
# endregion

class TacticalMomentumRankAlgorithm(QCAlgorithm):

    undesired_symbols_from_previous_deployment = []
    checked_symbols_from_previous_deployment = False

    def Initialize(self):
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2023, 6, 1)
        self.SetCash(1_000_000)
        
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        self.Settings.MinimumOrderMarginPortfolioPercentage = 0

        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Raw
        self.AddUniverseSelection(QQQConstituentsUniverseSelectionModel(self.UniverseSettings))
        
        self.AddAlpha(MomentumQuantilesAlphaModel(
            int(self.GetParameter("quantiles")),
            int(self.GetParameter("lookback_months"))
        ))

        self.Settings.RebalancePortfolioOnSecurityChanges = False
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.day = -1
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel(self.rebalance_func))

        self.AddRiskManagement(NullRiskManagementModel())

        self.SetExecution(ImmediateExecutionModel())

        self.SetWarmUp(timedelta(7))

    def rebalance_func(self, time):
        if self.day != self.Time.day and not self.IsWarmingUp and self.CurrentSlice.QuoteBars.Count > 0:
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


