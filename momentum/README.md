## Multi-Asset Momentum Strategy

This is a trend following strategy that has three specific parts

### 1 - Multi-Timeframe Trend

The momentum signal consists of three distinct time horizons:

- Fast (1-month): Captures short-term market momentum and recent price movements
- Medium (3-month): Identifies intermediate trends, historically the most persistent
- Slow (12-month): Captures longer-term directional moves while avoiding very long-term reversals

The strategy combines these three timeframes because trends happen at different speeds. A stock might be in a short-term dip but still part of a longer-term uptrend.  

### 2 - Position Sizing

Less volatile assets have larger positions and more volatile assets have smaller positions. Risk is allocated equally across sectors.

### 3 - Risk Management

Portfolio volatility targeting tries to keep things steady while buffers prevent constant tweaking (which also decreases transaction costs). Finally maximum position sizing stops any one position from becoming too large and killing the entire portfolio.

### Performance

![Results](myplot.png)

### Improvements

- Alternative Universe Construction: Addition of sector ETFs, international fixed income, or alternative risk premia
- Transaction Cost Optimization: More sophisticated rebalancing rules to minimize costs while preserving alpha
