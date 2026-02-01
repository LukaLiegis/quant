## Equities Returns Prediction
I know it is called EQUITIES returns prediction, but getting a LOT of intraday data for crypto is free whereas getting the same data for equities is not, but the general concept can still be applied.

The general idea with this project is:

1. We get a signal for each asset using all assets as features, to get the signal we use some mega cool ML technique (currently just XGBoost)
2. We then optimize this cool ML technique using Bayesian Optimization to squeeze out some extra returns.
3. The predicted value from this cool ML technique are then adjusted to take into account fees.
4. After fees are taken into account a z-score is calculated using predicted returns.
5. Now given the z-score and a covariance matrix (that was calculated in some step before) I can size positions for market neutrality and set some vol target.
6. For extra math skill show off i can also shrink the covariance matrix for more stability.
7. Hope that between steps 2. and 6. there is some sharpe left in the signal.

For now as we can see the stonk goes up. I still need to implement trade sizing with vol targeting and all the fee adjustments.
![strategy_returns.png](plots/strategy_returns.png)