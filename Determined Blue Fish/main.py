# region imports
from AlgorithmImports import *
# endregion

class Prueba070224(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2024, 2, 1)
        self.SetCash(100000)
        # Lista de futuros de commodities para suscribir
        commodities = [
            Futures.Metals.Gold,
            Futures.Energies.CrudeOilWTI,
            Futures.Softs.Sugar11,
            Futures.Grains.Wheat,
            Futures.Metals.Silver,
            Futures.Energies.NaturalGas,
            # Añade aquí otros commodities de interés
        ]

        # Suscribirse a cada commodity de la lista
        for commodity in commodities:
            future = self.AddFuture(commodity)

            # Configurar un filtro de contratos de futuros para obtener el contrato principal
            # Aquí se podría ajustar el filtro para optimizar la selección de contratos
            future.SetFilter(lambda x: x.Expiry(TimeSpan.FromDays(0), TimeSpan.FromDays(180)))

            # Configurar DataNormalizationMode (opcional)
            future.SetDataNormalizationMode(DataNormalizationMode.Adjusted)

            # Configurar la resolución de los datos
            future.Resolution = Resolution.Minute

    def OnData(self, data: Slice):
        pass
