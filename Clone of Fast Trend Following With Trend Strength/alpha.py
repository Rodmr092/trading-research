#region imports
from AlgorithmImports import *

from utils import GetPositionSize
from futures import categories
#endregion

class FastTrendFollowingLongAndShortWithTrendStrenthAlphaModel(AlphaModel):

    futures = []
    BUSINESS_DAYS_IN_YEAR = 256
    FORECAST_SCALAR_BY_SPAN = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1} # Given by author on https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/chapter7.py

    def __init__(self, algorithm, slow_ema_span, abs_forecast_cap, sigma_span, target_risk, blend_years):
        self.algorithm = algorithm
        self.slow_ema_span = slow_ema_span
        self.fast_ema_span = int(self.slow_ema_span / 4)                  # "Any ratio between the two moving average lengths of two and six gives statistically indistinguishable results." (p.165)
        self.annulaization_factor = self.BUSINESS_DAYS_IN_YEAR ** 0.5

        self.abs_forecast_cap = abs_forecast_cap
        
        self.sigma_span = sigma_span
        self.target_risk = target_risk
        self.blend_years = blend_years

        self.idm = 1.5                                                    # Instrument Diversification Multiplier. Hardcoded in https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/chapter8.py
        self.forecast_scalar = self.FORECAST_SCALAR_BY_SPAN[self.fast_ema_span] 

        self.categories = categories
        self.total_lookback = timedelta(365*self.blend_years+self.slow_ema_span)

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
            risk_adjusted_ewmac = future.ewmac.Current.Value / daily_risk_price_terms
            scaled_forecast_for_ewmac = risk_adjusted_ewmac * self.forecast_scalar 
            forecast = max(min(scaled_forecast_for_ewmac, self.abs_forecast_cap), -self.abs_forecast_cap)

            if forecast * position == 0:
                continue
            # Save some data for the PCM
            current_contract.forecast = forecast
            current_contract.position = position

            # Create the insights
            local_time = Extensions.ConvertTo(algorithm.Time, algorithm.TimeZone, future.Exchange.TimeZone)
            expiry = future.Exchange.Hours.GetNextMarketOpen(local_time, False) - timedelta(seconds=1)
            insights.append(Insight.Price(future.Mapped, expiry, InsightDirection.Up if forecast * position > 0 else InsightDirection.Down))
        
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

            if security.Symbol.IsCanonical():
                # Add some members to track price history
                security.adjusted_history = pd.Series()
                security.raw_history = pd.Series()
                
                # Create indicators for the continuous contract
                security.fast_ema = algorithm.EMA(security.Symbol, self.fast_ema_span, Resolution.Daily)
                security.slow_ema = algorithm.EMA(security.Symbol, self.slow_ema_span, Resolution.Daily)
                security.ewmac = IndicatorExtensions.Minus(security.fast_ema, security.slow_ema)

                security.automatic_indicators = [security.fast_ema, security.slow_ema]

                self.futures.append(security)

        for security in changes.RemovedSecurities:
            # Remove consolidator + indicators
            algorithm.SubscriptionManager.RemoveConsolidator(security.Symbol, security.consolidator)
            if security.Symbol.IsCanonical():
                for indicator in security.automatic_indicators:
                    algorithm.DeregisterIndicator(indicator)

