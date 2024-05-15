# region imports
from AlgorithmImports import *
from universe import FrontMonthFutureUniverseSelectionModel
from portfolio import InverseVolatilityPortfolioConstructionModel
# endregion

class InverseVolatilityRankAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2022, 12, 31)
        self.SetCash(100000000)     # For a large future universe, the fund needed would be large

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        self.UniverseSettings.ExtendedMarketHours = True
        self.UniverseSettings.Resolution = Resolution.Minute
        
        # Seed initial price data
        self.SetSecurityInitializer(BrokerageModelSecurityInitializer(
            self.BrokerageModel, FuncSecuritySeeder(self.GetLastKnownPrices)))

        # We only want front month contract
        self.AddUniverseSelection(FrontMonthFutureUniverseSelectionModel(self.SelectFutureChainSymbols))

        # Since we're using all assets for portfolio optimization, we emit constant alpha for every security
        self.AddAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(7)))

        # A custom PCM to size by inverse volatility
        self.SetPortfolioConstruction(InverseVolatilityPortfolioConstructionModel())

        self.SetWarmup(31, Resolution.Daily)

    def SelectFutureChainSymbols(self, utcTime):
        return [
            Symbol.Create(Futures.Indices.VIX, SecurityType.Future, Market.CFE),
            Symbol.Create(Futures.Indices.SP500EMini, SecurityType.Future, Market.CME),
            Symbol.Create(Futures.Indices.NASDAQ100EMini, SecurityType.Future, Market.CME),
            Symbol.Create(Futures.Indices.Dow30EMini, SecurityType.Future, Market.CME),
            Symbol.Create(Futures.Energies.BrentCrude, SecurityType.Future, Market.NYMEX),
            Symbol.Create(Futures.Energies.Gasoline, SecurityType.Future, Market.NYMEX),
            Symbol.Create(Futures.Energies.HeatingOil, SecurityType.Future, Market.NYMEX),
            Symbol.Create(Futures.Energies.NaturalGas, SecurityType.Future, Market.NYMEX),
            Symbol.Create(Futures.Grains.Corn, SecurityType.Future, Market.CBOT),
            Symbol.Create(Futures.Grains.Oats, SecurityType.Future, Market.CBOT),
            Symbol.Create(Futures.Grains.Soybeans, SecurityType.Future, Market.CBOT),
            Symbol.Create(Futures.Grains.Wheat, SecurityType.Future, Market.CBOT),
        ]
