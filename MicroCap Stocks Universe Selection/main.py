# region imports
from AlgorithmImports import *
# endregion
#Se filtran los 200 stocks más líquidos, y luego se invierte en aquellos 10 stocks con el MarketCap más pequeño
class smallcapstocks(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        self.rebalanceTime = datetime.min
        self.activeStocks = set()

        self.AddUniverse(self.CoarseFilter, self.FineFilter)
        self.UniverseSettings.Resolution = Resolution.Hour

        self.portfolioTargets = []

    def CoarseFilter(self, coarse):
        if self.Time <= self.rebalanceTime:
            return self.Universe.Unchanged
        #Si no regresamos, se ajusta el nuevo rebalance time para checar el siguiente mes...
        self.rebalanceTime = self.Time + timedelta(30)
        sortedbyDollarVolume = sorted(coarse, key = lambda x: x.DollarVolume, reverse = True)
        return [x.Symbol for x in sortedbyDollarVolume if x.Price > 10 and x.HasFundamentalData][:200]

    def FineFilter(self, fine):
        sortedbyPE = sorted(fine, key=lambda x: x.MarketCap)
        return [x.Symbol for x in sortedbyPE if x.MarketCap > 0][:10]

    def OnSecuritiesChanged(self, changes):
        for x in changes.RemovedSecurities:
            self.Liquidate(x.Symbol)
            self.activeStocks.remove(x.Symbol)

        for x in changes.AddedSecurities:
            self.activeStocks.add(x.Symbol)

        self.portfolioTargets = [PortfolioTarget(symbol, 1/len(self.activeStocks))
                                    for symbol in self.activeStocks]


    def OnData(self, data: Slice):
        if self.portfolioTargets == []:
            return
        for symbol in self.activeStocks:
            if symbol not in data:
                return

        self.SetHoldings(self.portfolioTargets)
        self.portfolioTargets = []

