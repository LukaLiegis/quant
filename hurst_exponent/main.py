import numpy as np
import yfinance as yf
from scipy import stats


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


def calculate_stylized_facts(returns):
    returns = np.array(returns)
    abs_returns = np.abs(returns)
    sq_returns = np.square(returns)

    facts = {}

    facts['mean'] = np.mean(returns)
    facts['variance'] = np.var(returns, ddof=1)
    facts['std'] = np.std(returns, ddof=1)
    facts['skew'] = stats.skew(returns)
    facts['kurtosis'] = stats.kurtosis(returns, fisher=False)

    max_lag = min(40, len(returns) // 4)
    facts['acf_returns'] = []
    facts['acf_abs_returns'] = []
    facts['acf_sq_returns'] = []

    for h in range(1, max_lag + 1):
        if len(returns) > h:
            acf_ret = np.corrcoef(returns[:-h], returns[h:])[0, 1]
            facts['acf_returns'].append(acf_ret if not np.isnan(acf_ret) else 0)

            acf_abs = np.corrcoef(abs_returns[:-h], abs_returns[h:])[0, 1]
            facts['acf_abs_returns'].append(acf_abs if not np.isnan(acf_abs) else 0)

            acf_sq = np.corrcoef(sq_returns[:-h], sq_returns[h:])[0, 1]
            facts['acf_sq_returns'].append(acf_sq if not np.isnan(acf_sq) else 0)

    return facts


if __name__ == '__main__':
    ...