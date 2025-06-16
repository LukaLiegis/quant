# quant

I learn best by doing rather than just be reading and I want to learn everything there is to know in the field of quant research, so this is my attempt of re-creating and testing as many different aspects of the finance world. Each project has a write-up attached inside the folder.

### beta_calculation
A comparison of multiple research papers on ways of calculating an asset's beta. 

The methods are compared by using RMSE.

### momentum
A systematic time series momentum strategy with volatility adjusted position sizing across all asset classes. 

### waymo_long
A not so serious approach of trying to decompose Google's stock price and go long waymo by removing all of Google's other businesses from its stock price.

### calculating_vol

Implementing the six different methods of calculating volatility.

### volatility_forecasting
Comparing different volatility forecasting methods (HAR, Kernel Ridge) from research papers. A single way is used to calculate volatility and the methods are compared to a standard GARCH model to see how each model's forecasting ability differs.

### yield_curve_rv 
A backtest of a yield curve relative value trading strategy that is based on PCA decomposition of the yield curve.

### stock_prediction 
A recreation of [this](https://github.com/borisbanushev/stockpredictionai) github project which was one of the first quant projects I had ever looked at.

The script uses a variational autoencoder to create features from the returns dataset, and then a Wasserstein GAN to forecast day ahead returns. Bayesian optimisation is used to tune the GAN's hyperparameters.