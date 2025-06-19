import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

YIELD_COLUMNS = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2',
                'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
MATURITY_LABELS = ['1M', '3M', '6M', '1Y', '2Y', '3Y',
                  '5Y', '7Y', '10Y', '20Y', '30Y']
MATURITY_YEARS = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]


def load_and_prepare_data():
    df = pd.read_csv('combined_curves.csv')
    df['observation_date'] = pd.to_datetime(df['observation_date'])

    yields = df[YIELD_COLUMNS].apply(pd.to_numeric, errors='coerce')
    yields = yields.dropna()
    yields_change = yields.diff().dropna()

    return df, yields, yields_change


def perform_pca_analysis(
        yield_changes: pd.DataFrame,
        yield_levels: pd.DataFrame,
) -> Dict:
    pca = PCA()
    pca_result = pca.fit_transform(yield_changes)

    results = {
        'pca_model': pca,
        'components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'explained_variance': pca.explained_variance_,
        'scores': pca_result,
        'pc_std_devs': np.sqrt(pca.explained_variance_),
    }

    results['cumulative_variance'] = np.cumsum(results['explained_variance_ratio'])
    results['components_scaled'] = results['components'] * results['pc_std_devs'][:, np.newaxis]

    pca_levels = PCA()
    pca_results_levels = pca_levels.fit_transform(yield_levels)
    results['pc_levels'] = {
        'components': pca_levels.components_,
        'explained_variance_ratio': pca_levels.explained_variance_ratio_,
        'scores': pca_results_levels,
        'pc_std_devs': np.sqrt(pca_levels.explained_variance_),
    }

    return results


def plot_principal_components(
        pca_results: Dict
):
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 8))

    components = pca_results['components'][:3]
    explained_var = pca_results['explained_variance_ratio'][:3]
    pc_std_devs = pca_results['pc_std_devs'][:3]

    component_names = ['Level', 'Slope', 'Curvature']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(3):
        component_scaled = components[i] * pc_std_devs[i] * 100
        ax1.plot(MATURITY_YEARS, component_scaled, 'o-',
                 linewidth=2.5, markersize=6, color=colors[i],
                 label=f'PC{i + 1}: {component_names[i]} ({explained_var[i]:.1%})')

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_title('Principal Components (1-std dev moves)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Years to Maturity')
    ax1.set_ylabel('Yield Change (bp)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(MATURITY_YEARS)
    ax1.set_xticklabels(MATURITY_LABELS, rotation=45)
    ax1.legend()
    plt.show()


def main():

    df, yields, yield_changes = load_and_prepare_data()

    pca_results = perform_pca_analysis(yield_changes, yields)

    plot_principal_components(pca_results)

if __name__ == "__main__":
    main()