## Multi-Asset Momentum Strategy

This strategy implements a systematic time series momentum approach across four major asset classes (equities, bonds, commodities, currencies) based on [this](https://threadreaderapp.com/thread/1587591552691765251.html) twitter thread and the Time series momentum [paper](https://www.sciencedirect.com/science/article/pii/S0304405X11002613).   

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
