#region imports
from AlgorithmImports import *
#endregion

class BufferedPortfolioConstructionModel(EqualWeightingPortfolioConstructionModel):

    def __init__(self, rebalance, buffer_scaler):
        super().__init__(rebalance)
        self.buffer_scaler = buffer_scaler

    def CreateTargets(self, algorithm: QCAlgorithm, insights: List[Insight]) -> List[PortfolioTarget]:
        targets = super().CreateTargets(algorithm, insights)
        adj_targets = []
        for insight in insights:
            future_contract = algorithm.Securities[insight.Symbol]
            optimal_position = future_contract.forecast * future_contract.position / 10

            ## Create buffer zone to reduce churn
            buffer_width = self.buffer_scaler * abs(future_contract.position)
            upper_buffer = round(optimal_position + buffer_width)
            lower_buffer = round(optimal_position - buffer_width)
            
            # Determine quantity to put holdings into buffer zone
            current_holdings = future_contract.Holdings.Quantity
            if lower_buffer <= current_holdings <= upper_buffer:
                continue
            quantity = lower_buffer if current_holdings < lower_buffer else upper_buffer

            # Place trades
            adj_targets.append(PortfolioTarget(insight.Symbol, quantity))
        
        # Liquidate contracts that have an expired insight
        for target in targets:
            if target.Quantity == 0:
                adj_targets.append(target)

        return adj_targets
