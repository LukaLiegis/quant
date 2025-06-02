import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

target_ticker = 'GOOGL'

ad_universe = [
    'META',
    'AMZN',
    'AAPL',
    'NFLX',
    'DIS',
    'SNAP',
    'PINS',
    'ROKU',
    'TTD',
    'SPOT'
]

sector_etfs = [
    'XLC',
    'QQQ',
    'VGT'
]

european_tickers = [
    'WBD',
    'PARA'
]

hedge_universe = ad_universe + sector_etfs + european_tickers

all_tickers = hedge_universe + [target_ticker]

data = yf.download(all_tickers, start='2018-01-01', end='2025-05-01')['Close']

returns = data.pct_change().dropna()
target_returns = returns[target_ticker]


def correlation():
    """
    Correlation coefficient.
    """
    correlations = []

    for ticker in hedge_universe:
        if ticker in returns.columns:
            corr = target_returns.corr(returns[ticker])
            correlations.append({'Ticker': ticker, 'Correlation': corr})

    corr_df = pd.DataFrame(correlations).sort_values(by='Correlation', ascending=False)

    plt.figure(figsize = (12, 8))
    sns.barplot(y='Ticker', x='Correlation', data=corr_df)
    plt.title('Correlation')
    plt.show()

    return corr_df


def multivariate_regression():
    """
    multivariate regression
    """
    hedge_tickers = ['META', 'XLC', 'AMZN', 'NFLX', 'QQQ']
    available_hedges = [t for t in hedge_tickers if t in returns.columns]

    y = returns[target_ticker].values
    X = returns[available_hedges].values

    reg = LinearRegression().fit(X, y)

    hedge_ratios = - reg.coef_

    results = pd.DataFrame({
        'Ticker': available_hedges,
        'Hedge_Ratio': hedge_ratios,
        'Position': ['Short' if hr > 0 else 'Long' for hr in hedge_ratios],
    })

    hedge_returns = np.sum(hedge_ratios.reshape(1, -1) * returns[available_hedges].values, axis=1)
    hedged_portfolio = target_returns.values + hedge_returns

    original_vol = np.std(target_returns) * np.sqrt(252)
    hedged_vol = np.std(hedged_portfolio) * np.sqrt(252)
    vol_reduction = (original_vol - hedged_vol) / original_vol * 100

    print(f"Original GOOGL Volatility: {original_vol:.2%}")
    print(f"Hedged Portfolio Volatility: {hedged_vol:.2%}")
    print(f"Volatility Reduction: {vol_reduction:.1f}%")
    print(f"R-squared: {reg.score(X, y):.3f}")

    return {
        'hedge_ratios': results,
        'hedge_returns': hedge_returns,
        'original_returns': target_returns.values,
        'r_squared': reg.score(X, y),
        'vol_reduction': vol_reduction,
        'dates': target_returns.index,
    }

def pca_hedge():
    """
    PCA Method
    """
    ad_tickers = [t for t in ad_universe if t in returns.columns]

    ad_returns = returns[ad_tickers].dropna()
    target_returns_pca = returns[target_ticker].reindex(ad_returns.index)

    scaler = StandardScaler()
    ad_returns_scaled = scaler.fit_transform(ad_returns)

    pca = PCA(n_components = 3)
    principal_components = pca.fit_transform(ad_returns_scaled)

    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f'PC{i + 1}: {var:.3f}')
    print(f'Total explaind variance: {pca.explained_variance_ratio_.sum():.3f}')

    pca_reg = LinearRegression().fit(principal_components, target_returns_pca)

    pc_loadings = pca.components_.T
    hedge_weights = -np.dot(pc_loadings, pca_reg.coef_)

    hedge_weights_normalized = hedge_weights / np.sum(np.abs(hedge_weights))

    pca_hedge_portfolio = np.sum(hedge_weights_normalized.reshape(1, -1) * ad_returns.values, axis=1)

    pca_hedge_returns = target_returns_pca.values + pca_hedge_portfolio

    original_vol = np.std(target_returns_pca) * np.sqrt(252)
    pca_hedged_vol = np.std(pca_hedge_returns) * np.sqrt(252)
    pca_vol_reduction = (original_vol - pca_hedged_vol) / original_vol * 100

    print(f"Original GOOGL Volatility: {original_vol:.2%}")
    print(f"PCA Hedged Volatility: {pca_hedged_vol:.2%}")
    print(f"Volatility Reduction: {pca_vol_reduction:.1f}%")
    print(f"R-squared: {pca_reg.score(principal_components, target_returns_pca):.3f}")

    return {
        'hedge_weights': hedge_weights,
        'hedge_returns': pca_hedge_returns,
        'original_returns': target_returns_pca.values,
        'explained_variance': pca.explained_variance_ratio_.sum(),
        'vol_reduction': pca_vol_reduction,
        'dates': target_returns_pca.index,
    }


def beta_hedge(hedge_ticker:str = 'XLC', window:int = 252):
    """
    Time varying beta hedge
    """
    target_returns_beta = returns[target_ticker]
    hedge_returns = returns[hedge_ticker]

    rolling_beta = []
    rolling_r_squared = []
    dates = []

    for i in range(window, len(target_returns_beta)):
        y = target_returns_beta.iloc[i - window: i].values
        x = hedge_returns.iloc[i - window: i].values

        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < window * 0.8:
            rolling_beta.append(np.nan)
            rolling_r_squared.append(np.nan)
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            rolling_beta.append(slope)
            rolling_r_squared.append(r_value ** 2)

        dates.append(target_returns_beta.index[i])

    rolling_stats = pd.DataFrame({
        'Date': dates,
        'Beta': rolling_beta,
        'R_squared': rolling_r_squared,
    }).set_index('Date')

    tv_hedge_returns = []
    for i, (date, beta) in enumerate(rolling_stats['Beta'].items()):
        if not np.isnan(beta):
            target_ret = target_returns_beta.loc[date]
            hedge_ret = hedge_returns.loc[date]

            hedge_ret = target_ret - beta * hedge_ret
            tv_hedge_returns.append(hedge_ret)
        else:
            tv_hedge_returns.append(np.nan)

    rolling_stats['Hedged_Returns'] = tv_hedge_returns
    rolling_stats['Original_Returns'] = target_returns_beta.loc[rolling_stats.index]

    hedge_vol = np.std(rolling_stats['Hedged_Returns'].dropna()) * np.sqrt(252)
    original_vol = np.std(rolling_stats['Original_Returns'].dropna()) * np.sqrt(252)
    vol_reduction = (original_vol - hedge_vol) / original_vol * 100

    # Plot rolling beta
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 10))

    rolling_stats['Beta'].plot(ax = ax1, title = f'Rolling Beta: {target_ticker} vs {hedge_ticker}')
    ax1.axhline(y = rolling_stats['Beta'].mean(), color = 'r', linestyle = '--',
               label = f'Mean Beta: {rolling_stats["Beta"].mean():.3f}')
    ax1.legend()
    ax1.grid(True)

    rolling_stats['R_squared'].plot(ax = ax2, title = 'Rolling R^2', color = 'orange')
    ax2.axhline(y=rolling_stats['R_squared'].mean(), color='r', linestyle='--',
                label=f'Mean R²: {rolling_stats['R_squared'].mean():.3f}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


    current_beta = rolling_stats['Beta'].dropna().iloc[-1]
    hedge_ratio = -current_beta

    print(f"Current Beta: {current_beta:.3f}")
    print(f"Recommended Hedge Ratio: {hedge_ratio:.3f}")
    print(f"Average Beta: {rolling_stats['Beta'].mean():.3f}")
    print(f"Beta Volatility: {rolling_stats['Beta'].std():.3f}")
    print(f"Average R-Squared: {rolling_stats['R_squared'].mean():.3f}")

    return {
        'rolling_stats': rolling_stats,
        'current_beta': current_beta,
        'hedge_ratio': hedge_ratio,
        'beta_vol': rolling_stats['Beta'].std(),
        'vol_reduction': vol_reduction
    }


def plot_returns_comparison(mv_results, pca_results, tv_results):
    """
    Plot returns comparison for the three strategies
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 12))

    ax1.plot(mv_results['dates'], mv_results['original_returns'],
             label='Original GOOGL', alpha=0.7, linewidth=0.5)
    ax1.plot(mv_results['dates'], mv_results['hedge_returns'],
             label='Multivariate Hedge', alpha=0.7, linewidth=0.5)
    ax1.plot(pca_results['dates'], pca_results['hedge_returns'],
             label='PCA Hedge', alpha=0.7, linewidth=0.5)
    ax1.plot(tv_results['rolling_stats'].index,
             tv_results['rolling_stats']['Hedged_Returns'],
             label='Time-Varying Beta Hedge', alpha=0.7, linewidth=0.5)

    ax1.set_title('Daily Returns Comparison', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    original_cumret = (1 + pd.Series(mv_results['original_returns'],
                                     index=mv_results['dates'])).cumprod() - 1
    mv_cumret = (1 + pd.Series(mv_results['hedge_returns'],
                               index=mv_results['dates'])).cumprod() - 1
    pca_cumret = (1 + pd.Series(pca_results['hedge_returns'],
                                index=pca_results['dates'])).cumprod() - 1
    tv_cumret = (1 + tv_results['rolling_stats']['Hedged_Returns'].dropna()).cumprod() - 1

    ax2.plot(original_cumret.index, original_cumret * 100,
             label='Original GOOGL', linewidth=2)
    ax2.plot(mv_cumret.index, mv_cumret * 100,
             label='Multivariate Hedge', linewidth=2)
    ax2.plot(pca_cumret.index, pca_cumret * 100,
             label='PCA Hedge', linewidth=2)
    ax2.plot(tv_cumret.index, tv_cumret * 100,
             label='Time-Varying Beta Hedge', linewidth=2)

    ax2.set_title('Cumulative Returns Comparison', fontsize=14)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Returns (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


corr_results = correlation()
print(corr_results)
mv_results = multivariate_regression()
pca_results = pca_hedge()
tv_results = beta_hedge()

comparison = pd.DataFrame({
    'Strategy': ['Multivariate Regression', 'PCA Hedge', 'Time-Varying Beta'],
    'Volatility Reduction': [
        mv_results['vol_reduction'],
        pca_results['vol_reduction'],
        np.nan
    ],
    'R_Squared': [
        mv_results['r_squared'],
        pca_results['explained_variance'],
        tv_results['rolling_stats']['R_squared'].mean()
    ],
    'Complexity': ['Medium', 'High', 'Low'],
    'Rebalancing Frequency': ['Monthly', 'Quarterly', 'Daily/Weekly'],
})

print(comparison.round(3))

plot_returns_comparison(mv_results, pca_results, tv_results)