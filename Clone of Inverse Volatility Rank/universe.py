#region imports
from AlgorithmImports import *
from Selection.FutureUniverseSelectionModel import FutureUniverseSelectionModel
#endregion

class FrontMonthFutureUniverseSelectionModel(FutureUniverseSelectionModel):
    '''Creates futures chain universes that select the front month contract and runs a user
    defined futureChainSymbolSelector every day to enable choosing different futures chains'''
    def __init__(self, select_future_chain_symbols, rebalancePeriod = 7):
        super().__init__(timedelta(rebalancePeriod), select_future_chain_symbols)

    def Filter(self, filter):
        '''Defines the futures chain universe filter'''
        return (filter.FrontMonth()
                      .OnlyApplyFilterAtMarketOpen())
