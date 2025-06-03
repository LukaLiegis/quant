import tqdm
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import timedelta
from scipy.optimize import curve_fit

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table', {'id': 'constituents'})

sp500_data = []

for row in table.find_all('tr')[1:]:
    cols = row.find_all('td')
    if len(cols) >= 4:
        ticker = cols[0].text.strip()
        company = cols[1].text.strip()
        sector = cols[3].text.strip()
        sub_sector = cols[4].text.strip() if len(cols) > 4 else ''

        ticker = ticker.replace('.', '-')

        sp500_data.append({
            'ticker': ticker,
            'company': company,
            'sector': sector,
            'sub_sector': sub_sector
        })

sp500_df = pd.DataFrame(sp500_data)

spx = yf.download('^GSPC')
spx_log_returns = pd.DataFrame({
    'date': spx.index,
    'spx_log_returns': np.log(spx['Close'] / spx['Close'].shift(1)),
}).dropna()

stock_data = []
failed_tickers = []

for ticker in tqdm(sp500_df['ticker'].tolist()):
    try:
        data = yf.download(ticker)
        if len(data) > 0:
            stock_data[ticker] = data['Close']
        else:
            failed_tickers.append(ticker)
    except Exception as e:
        failed_tickers.append(ticker)

stock_prices = pd.DataFrame(stock_data)

stock_log_returns = np.log(stock_prices / stock_prices.shift(1))


def calculate_rolling_stats(
        log_returns,
        window: int = 252,
        skip_window: int = 21
):
    shifted_returns = log_returns.shift(skip_window)

    rolling_sum = shifted_returns.rolling(window).sum()

    rolling_vol = shifted_returns.rolling(window).std() * np.sqrt(252)

    return rolling_sum, rolling_vol


def polynomial_function(x, *coeffs):
    return np.sum(c * x ** i for i, c in enumerate(coeffs))


def fit_polynomial_model(y_data, degree: int = 4):
    x_data = np.arange(len(y_data))

    mask = ~np.isnan(y_data)
    x_clean = x_data[mask]
    y_clean = y_data[mask]

    if len(y_clean) < degree + 1:
        return None, None

    try:
        p0 = [0.1] * (degree + 1)

        popt, _ = curve_fit(polynomial_function, x_clean, y_clean, p0=p0, maxfev=5000)

        fitted_values = polynomial_function(x_clean, *popt)

        return popt, fitted_values
    except Exception as e:
        print(e)
        return None, None

stock_rolling_sum , stock_rolling_vol = calculate_rolling_stats(stock_log_returns)

spx_rolling_sum = spx_log_returns.set_index('date')['spx_log_returns'].shift(21).rolling(252).sum()

relative_returns = pd.DataFrame()
volatility_scaled_returns = pd.DataFrame()

for ticker in spx_rolling_sum.index.columns:
    common_dates = stock_rolling_sum.index.intersection(spx_rolling_sum.index)
    rel_returns = stock_rolling_sum.loc[common_dates, ticker] - spx_rolling_sum.loc[common_dates]
    vol_scaled = rel_returns / stock_rolling_vol.loc[common_dates, ticker]

    relative_returns[ticker] = rel_returns
    volatility_scaled_returns[ticker] = vol_scaled

volatility_scaled_returns = volatility_scaled_returns.dropna()

cutoff_date = volatility_scaled_returns.index.max() - timedelta(days = 21)
volatility_scaled_returns_fit = volatility_scaled_returns[volatility_scaled_returns.index <= cutoff_date]

predictions = []

for ticker in tqdm(volatility_scaled_returns_fit.column):
    y_data = volatility_scaled_returns_fit[ticker].values

    if len(y_data[~np.isnan(y_data)]) < 5:
        continue

    coeffs, fitted_values = fit_polynomial_model(y_data, degree=4)

    if coeffs is not None:
        predicted_returns = fitted_values[-1]
        predictions.append({
            'ticker': ticker,
            'predicted_returns': predicted_returns,
        })

predictions_df = pd.DataFrame(predictions)

predictions_df['decile'] = pd.qcut(predictions_df['predicted_returns'], 10, labels=range(1, 11))

top_decile = predictions_df[predictions_df['decile'] == 10].sort_values(by='predicted_returns', ascending=False)
print(top_decile['ticker'].to_list())

watchlist = top_decile.merge(sp500_df, on='ticker', how='left')