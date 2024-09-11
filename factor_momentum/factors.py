import numpy as np
import pandas as pd

# Assume we have the following data:
# monthly_returns: DataFrame of monthly returns for each commodity
# futures_data: DataFrame with nearest (F1) and next-nearest (F2) futures prices
# cftc_data: DataFrame with CFTC positions data
# daily_returns: DataFrame of daily returns for each commodity
# inflation_data: DataFrame or Series of monthly inflation data
# open_interest_data: DataFrame of open interest for each commodity

def calculate_factors(monthly_returns, futures_data, cftc_data, daily_returns, inflation_data, open_interest_data):
    
    # 1. Average Commodity Factor (AVG)
    avg_commodity = monthly_returns.mean(axis=1)
    
    # 2. Momentum
    momentum = monthly_returns.rolling(window=12).sum().shift(1)
    
    # 3. Basis
    basis = (futures_data['F2'] - futures_data['F1']) / futures_data['F1']
    
    # 4. Basis-Momentum
    f1_momentum = futures_data['F1'].pct_change(12)
    f2_momentum = futures_data['F2'].pct_change(12)
    basis_momentum = f2_momentum - f1_momentum
    
    # 5. Hedging Pressure
    hedging_pressure = (cftc_data['Commercial Long'] - cftc_data['Commercial Short']) / \
                       (cftc_data['Commercial Long'] + cftc_data['Commercial Short'])
    
    # 6. Value
    value = -monthly_returns.rolling(window=60).sum()
    
    # 7. Skewness
    skewness = monthly_returns.rolling(window=12).skew()
    
    # 8. Inflation Beta
    def inflation_beta(returns, inflation):
        # First, calculate unexpected inflation (you might want to use a more sophisticated method)
        unexpected_inflation = inflation - inflation.rolling(window=12).mean()
        # Then calculate beta for each commodity
        betas = returns.apply(lambda x: np.cov(x, unexpected_inflation)[0,1] / np.var(unexpected_inflation))
        return betas
    
    inflation_betas = monthly_returns.rolling(window=60).apply(lambda x: inflation_beta(x, inflation_data))
    
    # 9. Volatility
    volatility = daily_returns.rolling(window=252).std() * np.sqrt(252)
    
    # 10. Open Interest
    open_interest = open_interest_data
    
    # Create long-short portfolios
    def long_short_portfolio(factor, n=3):
        ranks = factor.rank(axis=1, pct=True)
        longs = (ranks > (1 - 1/n)).astype(int)
        shorts = (ranks <= 1/n).astype(int)
        return (longs - shorts) / n
    
    factors = {
        'AVG': avg_commodity,
        'Momentum': long_short_portfolio(momentum),
        'Basis': long_short_portfolio(basis),
        'Basis-Momentum': long_short_portfolio(basis_momentum),
        'Hedging Pressure': long_short_portfolio(hedging_pressure),
        'Value': long_short_portfolio(value),
        'Skewness': long_short_portfolio(skewness),
        'Inflation Beta': long_short_portfolio(inflation_betas),
        'Volatility': long_short_portfolio(volatility),
        'Open Interest': long_short_portfolio(open_interest)
    }
    
    return factors