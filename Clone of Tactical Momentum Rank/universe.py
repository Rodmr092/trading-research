 #region imports
from AlgorithmImports import *
#endregion

class QQQConstituentsUniverseSelectionModel(ETFConstituentsUniverseSelectionModel):
    def __init__(self, universe_settings: UniverseSettings = None) -> None:
        symbol = Symbol.Create("QQQ", SecurityType.Equity, Market.USA)
        super().__init__(symbol, universe_settings, lambda constituents: [c.Symbol for c in constituents]) 
