# quant

I want to know everything possible that there is to know about markets, specifically the intersection between mathematics, computer science, and markets (some might call this Quantitative Research). To achieve this goal I aim to conduct as much research as possible into as many asset classes as possible in as many timeframes as possible. This does not only have to be alpha research, this could be portfolio optimization or optimal execution or time series generation. 

It is possible that some of these projects will not be complete, this is because while I think about the optimal approach to one project I may be setting up another or perhaps I am still just researching it. I will try to keep uncompleted projects to a minimum.

Each project contains a longer write-up that goes in-depth into all the inner workings.

### [Beta Calculation](./beta_calculation)
A comparison of multiple research papers on ways of calculating an asset's beta. 

The methods are compared by using RMSE.

### [Beta Compression](./beta_compression)

An implementation of the beta compression phenomenon from Frazzini and Pedersen.

### [Momentum](./momentum)
A systematic time series momentum strategy with volatility adjusted position sizing across all asset classes. 

### [Waymo Long](./waymo_long)
A not so serious approach of trying to decompose Google's stock price and go long waymo by removing all of Google's other businesses from its stock price.

### [Calculating Volatility](./calculating_vol)

Implementing the six different methods of calculating volatility for SPY and comparing this to the VIX index.

### [Volatility Forecasting](./volatility_forecasting)
Comparing different volatility forecasting methods (HAR, Kernel Ridge) from research papers. A single way is used to calculate volatility and the methods are compared to a standard GARCH model to see how each model's forecasting ability differs.

### [Yield Curve Decomposition](./yield_curve_decomp) 
This project initially started as a recreation of Salomon Smith Barney's paper by the name Principles of Principal Components which aims to decompose the yield curve into level slope and curvature since I did not want to go the backtest route due to data unavailability I chose to further analyze the PCs.

### [Stock Prediction AI](./stock_prediction) 
A recreation of [this](https://github.com/borisbanushev/stockpredictionai) github project which was one of the first quant projects I had ever looked at.

The script uses a variational autoencoder to create features from the returns dataset, and then a Wasserstein GAN to forecast day ahead returns. Bayesian optimisation is used to tune the GAN's hyperparameters.