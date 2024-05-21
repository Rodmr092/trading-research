#region imports
from AlgorithmImports import *

from utils import GetPositionSize
from futures import categories
#endregion

class AdjustedTrendAlphaModel(AlphaModel):

    futures = []
    BUSINESS_DAYS_IN_YEAR = 256
    FORECAST_SCALAR_BY_SPAN = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1} # Table 29 on page 177
    FDM_BY_RULE_COUNT = {1: 1.0, 2: 1.03, 3: 1.08, 4: 1.13, 5: 1.19, 6: 1.26}
    SCALE_AND_CAP_MAPPING_MULTIPLIER = 1.25 # Given on page 254

    def __init__(self, algorithm, emac_filters, abs_forecast_cap, sigma_span, target_risk, blend_years):
        self.algorithm = algorithm
        
        self.emac_spans = [2**x for x in range (1, emac_filters+1)]
        self.fast_ema_spans = self.emac_spans
        self.slow_ema_spans = [fast_span * 4 for fast_span in self.emac_spans] # "Any ratio between the two moving average lengths of two and six gives statistically indistinguishable results." (p.165)
        self.all_ema_spans = sorted(list(set(self.fast_ema_spans + self.slow_ema_spans)))

        self.annulaization_factor = self.BUSINESS_DAYS_IN_YEAR ** 0.5

        self.abs_forecast_cap = abs_forecast_cap
        
        self.sigma_span = sigma_span
        self.target_risk = target_risk
        self.blend_years = blend_years

        self.idm = 1.5                                                    # Instrument Diversification Multiplier. Hardcoded in https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/chapter8.py

        self.categories = categories
        self.total_lookback = timedelta(365*self.blend_years+self.all_ema_spans[-1])

        self.day = -1

    def Update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:        
        # Record the new contract in the continuous series
        if data.QuoteBars.Count:
            for future in self.futures:
                future.latest_mapped = future.Mapped

        # If warming up and still > 7 days before start date, don't do anything
        # We use a 7-day buffer so that the algorithm has active insights when warm-up ends
        if algorithm.StartDate - algorithm.Time > timedelta(7):
            return []
        
        if self.day == data.Time.day or data.Bars.Count == 0:
            return []

        # Estimate the standard deviation of % daily returns for each future
        sigma_pct_by_future = {}
        for future in self.futures:
            # Estimate the standard deviation of % daily returns
            sigma_pct = self.estimate_std_of_pct_returns(future.raw_history, future.adjusted_history)
            if sigma_pct is None:
                continue
            sigma_pct_by_future[future] = sigma_pct
        
        # Create insights
        insights = []
        weight_by_symbol = GetPositionSize({future.Symbol: self.categories[future.Symbol] for future in sigma_pct_by_future.keys()})
        for symbol, instrument_weight in weight_by_symbol.items():
            future = algorithm.Securities[symbol]
            current_contract = algorithm.Securities[future.Mapped]
            daily_risk_price_terms = sigma_pct_by_future[future] / (self.annulaization_factor) * current_contract.Price # "The price should be for the expiry date we currently hold (not the back-adjusted price)" (p.55)

            # Calculate target position
            position = (algorithm.Portfolio.TotalPortfolioValue * self.idm * instrument_weight * self.target_risk) \
                      /(future.SymbolProperties.ContractMultiplier * daily_risk_price_terms * (self.annulaization_factor))

            # Adjust target position based on forecast
            capped_forecast_by_span = {}
            for span in self.emac_spans:
                risk_adjusted_ewmac = future.ewmac_by_span[span].Current.Value / daily_risk_price_terms
                scaled_forecast_for_ewmac = risk_adjusted_ewmac * self.FORECAST_SCALAR_BY_SPAN[span]
                
                if span == 2: # "Double V" forecast mapping (page 253-254)
                    if scaled_forecast_for_ewmac < -20:
                        capped_forecast_by_span[span] = 0
                    elif -20 <= scaled_forecast_for_ewmac < -10:
                        capped_forecast_by_span[span] = -40 - (2 * scaled_forecast_for_ewmac)
                    elif -10 <= scaled_forecast_for_ewmac < 10:
                        capped_forecast_by_span[span] = 2 * scaled_forecast_for_ewmac
                    elif 10 <= scaled_forecast_for_ewmac < 20:
                        capped_forecast_by_span[span] = 40 - (2 * scaled_forecast_for_ewmac)
                    else:
                        capped_forecast_by_span[span] = 0
                
                elif span in [4, 64]: # "Scale and cap" forecast mapping
                    if scaled_forecast_for_ewmac < -15:
                        capped_forecast_by_span[span] = -15 * self.SCALE_AND_CAP_MAPPING_MULTIPLIER
                    elif -15 <= scaled_forecast_for_ewmac < 15:
                        capped_forecast_by_span[span] = scaled_forecast_for_ewmac * self.SCALE_AND_CAP_MAPPING_MULTIPLIER
                    else:
                        capped_forecast_by_span[span] = 15 * self.SCALE_AND_CAP_MAPPING_MULTIPLIER
                
                else: # Normal forecast capping
                    capped_forecast_by_span[span] = max(min(scaled_forecast_for_ewmac, self.abs_forecast_cap), -self.abs_forecast_cap)
            
            raw_combined_forecast = sum(capped_forecast_by_span.values()) / len(capped_forecast_by_span) # Calculate a weighted average of capped forecasts (p. 194)
            scaled_combined_forecast = raw_combined_forecast * self.FDM_BY_RULE_COUNT[len(capped_forecast_by_span)] # Apply a forecast diversification multiplier to keep the average forecast at 10 (p 193-194)
            capped_combined_forecast = max(min(scaled_combined_forecast, self.abs_forecast_cap), -self.abs_forecast_cap)

            if capped_combined_forecast * position == 0:
                continue
            # Save some data for the PCM
            current_contract.forecast = capped_combined_forecast
            current_contract.position = position
            
            # Create the insights
            local_time = Extensions.ConvertTo(algorithm.Time, algorithm.TimeZone, future.Exchange.TimeZone)
            expiry = future.Exchange.Hours.GetNextMarketOpen(local_time, False) - timedelta(seconds=1)
            insights.append(Insight.Price(future.Mapped, expiry, InsightDirection.Up if capped_combined_forecast * position > 0 else InsightDirection.Down))
        
        if insights:
            self.day = data.Time.day

        return insights

    def estimate_std_of_pct_returns(self, raw_history, adjusted_history):
        # Align history of raw and adjusted prices
        idx = sorted(list(set(adjusted_history.index).intersection(set(raw_history.index))))
        adjusted_history_aligned = adjusted_history.loc[idx]
        raw_history_aligned = raw_history.loc[idx]

        # Calculate exponentially weighted standard deviation of returns
        returns = adjusted_history_aligned.diff().dropna() / raw_history_aligned.shift(1).dropna() 
        rolling_ewmstd_pct_returns = returns.ewm(span=self.sigma_span, min_periods=self.sigma_span).std().dropna()
        if rolling_ewmstd_pct_returns.empty: # Not enough history
            return None
        # Annualize sigma estimate
        annulized_rolling_ewmstd_pct_returns = rolling_ewmstd_pct_returns * (self.annulaization_factor)
        # Blend the sigma estimate (p.80)
        blended_estimate = 0.3*annulized_rolling_ewmstd_pct_returns.mean() + 0.7*annulized_rolling_ewmstd_pct_returns.iloc[-1]
        return blended_estimate

    def consolidation_handler(self, sender: object, consolidated_bar: TradeBar) -> None:
        security = self.algorithm.Securities[consolidated_bar.Symbol]
        end_date = consolidated_bar.EndTime.date()
        if security.Symbol.IsCanonical():
            # Update adjusted history
            security.adjusted_history.loc[end_date] = consolidated_bar.Close
            security.adjusted_history = security.adjusted_history[security.adjusted_history.index >= end_date - self.total_lookback]
        else:
            # Update raw history
            continuous_contract = self.algorithm.Securities[security.Symbol.Canonical]
            if consolidated_bar.Symbol == continuous_contract.latest_mapped:
                continuous_contract.raw_history.loc[end_date] = consolidated_bar.Close
                continuous_contract.raw_history = continuous_contract.raw_history[continuous_contract.raw_history.index >= end_date - self.total_lookback]

    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        for security in changes.AddedSecurities:
            symbol = security.Symbol

            # Create a consolidator to update the history
            security.consolidator = TradeBarConsolidator(timedelta(1))
            security.consolidator.DataConsolidated += self.consolidation_handler
            algorithm.SubscriptionManager.AddConsolidator(symbol, security.consolidator)

            if symbol.IsCanonical():
                # Add some members to track price history
                security.adjusted_history = pd.Series()
                security.raw_history = pd.Series()

                # Create indicators for the continuous contract
                ema_by_span = {span: algorithm.EMA(symbol, span, Resolution.Daily) for span in self.all_ema_spans}
                security.ewmac_by_span = {}
                for i, fast_span in enumerate(self.emac_spans):
                    security.ewmac_by_span[fast_span] = IndicatorExtensions.Minus(ema_by_span[fast_span], ema_by_span[self.slow_ema_spans[i]])

                security.automatic_indicators = ema_by_span.values()

                self.futures.append(security)

        for security in changes.RemovedSecurities:
            # Remove consolidator + indicators
            algorithm.SubscriptionManager.RemoveConsolidator(security.Symbol, security.consolidator)
            if security.Symbol.IsCanonical():
                for indicator in security.automatic_indicators:
                    algorithm.DeregisterIndicator(indicator)
