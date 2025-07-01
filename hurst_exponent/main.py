import numpy as np
import yfinance as yf


def get_data(symbols_dict: dict) -> dict:
    data = {}

    for symbol, name in symbols_dict.items():
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period='max')

            if len(hist_data) > 0:
                prices = hist_data['Close'].dropna()
                returns = np.log(prices / prices.shift(1)).dropna()

                data[name] = {
                    'prices': prices,
                    'returns': returns,
                    'symbol': symbol,
                }
            else:
                print(f'No data for {symbol}')
        except Exception as e:
            print(f'Error: {e} for {symbol}')

    return data
