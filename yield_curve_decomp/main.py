

import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis

YIELD_COLUMNS = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2',
                'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
MATURITY_LABELS = ['1M', '3M', '6M', '1Y', '2Y', '3Y',
                  '5Y', '7Y', '10Y', '20Y', '30Y']
MATURITY_YEARS = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]


def load_and_prepare_data():
    df = pd.read_csv('combined_curves.csv')
    df['observation_date'] = pd.to_datetime(df['observation_date'])

    yields = df[YIELD_COLUMNS].apply(pd.to_numeric, errors='coerce')
    yields_change = yields.diff().dropna()

    return df, yields, yields_change


def perform_pca_analysis(
        yield_changes
):
    pca = PCA()
    pca_result = pca.fit_transform(yield_changes)

    components = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance = pca.explained_variance_

    pc_std_devs = np.sqrt(explained_variance)

    print(f"{'Component':<12} {'Std Dev (bp)':<15} {'Proportion':<12} {'Cumulative':<12}")

    cumulative = 0
    for i in range(min(3, len(explained_variance_ratio))):
        cumulative += explained_variance_ratio[i]
        print(f"PC #{i + 1:<8} {pc_std_devs[i] * 100:<15.2f} {explained_variance_ratio[i]:<12.1%} {cumulative:<12.1%}")

    print(f"\nFirst three components explain: {explained_variance_ratio[:3].sum():.1%} of total variance")

    return pca, components, explained_variance_ratio, explained_variance, pc_std_devs, pca_result


def factor_analysis(
        data
) -> Dict:
    results = {}

    for n_factors in range(3, 6):
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(data)

        log_likelihood = fa.score(data)
        results[n_factors] = {
            'model': fa,
            'log_likelihood': log_likelihood,
            'components': fa.components_,
        }

        print(f'{n_factors} components explained: {log_likelihood:.2f}')

    best_n_factors = max(results.keys(), key=(lambda k: results[k]['log_likelihood']))
    print('Best number of factors:', best_n_factors)

    return results


def analyze_time_series(
        pca_results,
):
    scores = pca_results['scores']

    cumulative_scores = np.cumsum(scores, axis=0)

    for i in range(3):
        pc_series = scores[:, i]
        print(f"PC{i + 1} - Mean: {pc_series.mean():.4f}, Std: {pc_series.std():.4f}")
        print(f"      Min: {pc_series.min():.4f}, Max: {pc_series.max():.4f}")
        print(f"      Skewness: {pd.Series(pc_series).skew():.3f}, Kurtosis: {pd.Series(pc_series).kurtosis():.3f}")

    return cumulative_scores


def plot_principal_components(
        pca_results
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    colors = ['blue', 'red', 'green']
    component_names = ['Level', 'Slope', 'Curvature']

    components = pca_results['components']
    explained_var = pca_results['explained_variance_ratio']

    for i in range(3):
        component_scaled = components[i] * np.sqrt(pca_results['explained_variance'][i]) * 100
        ax.plot(MATURITY_YEARS, component_scaled, 'o-',
                linewidth=2, markersize=6, color=colors[i],
                label=f'PC #{i + 1}: {component_names[i]} ({explained_var[i]:.1%})')

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_title('Principal Components of Yield Curve Changes',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Years to Maturity')
    ax.set_ylabel('Yield Change (bp)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(MATURITY_YEARS)
    ax.set_xticklabels(MATURITY_LABELS)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()


def analyze_yield_changes_statistics(
        yield_changes,
        maturity_labels
):
    stats = yield_changes.describe()
    print("\nSummary Statistics (in basis points):")
    print((stats * 100).round(2))

    print("\nCorrelation Matrix of Yield Changes:")
    corr_matrix = yield_changes.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=maturity_labels, yticklabels=maturity_labels)
    plt.title('Correlation Matrix of Yield Changes')
    plt.tight_layout()
    plt.show()

    return corr_matrix


def plot_variance_explained(
        explained_variance_ratio
):
    cumulative_variance = np.cumsum(explained_variance_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Individual Variance Explained by Each PC')
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 'o-', linewidth=2)
    ax2.axhline(y=99, color='red', linestyle='--', alpha=0.7, label='99% threshold')
    ax2.set_xlabel('Number of Principal Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Variance Explained')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():

    df, yields, yield_changes = load_and_prepare_data()

    pca, components, explained_variance_ratio, explained_variance, pc_std_devs, pca_result = perform_pca_analysis(
        yield_changes)

    fa_results = factor_analysis(yield_changes)

    corr_matrix = analyze_yield_changes_statistics(yield_changes, MATURITY_LABELS)

    plot_principal_components(pca_result)
    plot_variance_explained(explained_variance_ratio)

if __name__ == "__main__":
    main()