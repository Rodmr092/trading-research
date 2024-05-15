# region imports
from AlgorithmImports import *
# endregion

class ETFTrailingStop(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(100000)
        self.qqq = self.AddEquity("QQQ", Resolution.Hour).Symbol
        self.entryTicket = None
        self.stopmarketticket = None
        self.entryTime = datetime.min
        self.stopMarketOrderFillTime = datetime.min
        self.highestprice = 0

    def OnData(self, data: Slice):
        if (self.Time - self.stopMarketOrderFillTime).days < 30:
            return
        #Adquirimos el precio del activo
        price = self.Securities[self.qqq].Price
        
        #Limit Order
        if not self.Portfolio.Invested and not self.Transactions.GetOpenOrders(self.qqq):
            quantity = self.CalculateOrderQuantity(self.qqq, 0.9)
            self.entryTicket = self.LimitOrder(self.qqq, quantity, price, "Entry Order")
            self.entryTime = self.Time

        #Ajustar el precio de entrada si no se completa la orden en 1 día
        if (self.Time - self.entryTime).days > 1 and self.entryTicket.Status != OrderStatus.Filled:
            self.entryTime = self.Time
            updateFields = UpdateOrderFields()
            updateFields.LimitPrice = price

            self.entryTicket.Update(updateFields)

        #Mover el trailing stop price
        if self.stopmarketticket is not None and self.Portfolio.Invested:
            if price > self.highestprice:
                self.highestprice = price
                updateFields = UpdateOrderFields()
                updateFields.StopPrice = price * 0.95
                self.stopmarketticket.Update(updateFields)

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status != OrderStatus.Filled:
            return

        # Send stop loss if the limit order is completed
        if self.entryTicket is not None and self.entryTicket.OrderId == orderEvent.OrderId:
            self.stopmarketticket = self.StopMarketOrder(self.qqq, -self.entryTicket.Quantity, 0.95 * self.entryTicket.AverageFillPrice)

        # Save the fill time of the stop loss to avoid entering the market prematurely
        if self.stopmarketticket is not None and self.stopmarketticket.OrderId == orderEvent.OrderId:
            self.stopMarketOrderFillTime = self.Time
            self.highestprice = 0

