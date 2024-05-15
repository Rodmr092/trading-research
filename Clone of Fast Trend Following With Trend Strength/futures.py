# region imports
from AlgorithmImports import *
# endregion

categories = {
    Symbol.Create(Futures.Financials.Y10TreasuryNote, SecurityType.Future, Market.CBOT): ("Fixed Income", "Bonds"),
    Symbol.Create(Futures.Indices.SP500EMini, SecurityType.Future, Market.CME): ("Equity", "US")
}
