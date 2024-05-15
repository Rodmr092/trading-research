# region imports
from AlgorithmImports import *
# endregion

class algoritmoprueba(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        self.futureSymbols = ["ES", "LE", "ZW", "CL", "GC", "HE", "KC", "SB", "GF", "ZS", "ZC", "NG", "SI", "HG", "NQ", "CT"]
        for symbol in self.futureSymbols:
            future = self.AddFuture(symbol, Resolution.Daily)
            future.SetFilter(5, 180)
            self.Log(f"Suscrito a futuro: {symbol}")
        self.lastSelectedContracts = {}

    def OnData(self, slice):
        newSelections = {}
        
        for kvp in slice.FutureChains:
            chain = kvp.Value
            symbol = kvp.Key
            max_volume = 0
            contract_with_max_volume = None

            for contract in chain.Contracts.Values:
              # Verificar si este contrato no es el último seleccionado y si tiene el mayor volumen
              if (symbol not in self.lastSelectedContracts or contract.Symbol != self.lastSelectedContracts[symbol]) and contract.Volume > max_volume:
                max_volume = contract.Volume
                contract_with_max_volume = contract
            
            if contract_with_max_volume is not None:
                newSelections[symbol] = contract_with_max_volume
                self.Log(f"El contrato con más volumen es: {contract_with_max_volume.Symbol} de {symbol} con volumen: {max_volume}")

                
            # Actualizar las selecciones y realizar acciones de trading si es necesario
            for symbol, contract in newSelections.items():
                self.lastSelectedContracts[symbol] = contract.Symbol
