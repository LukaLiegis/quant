import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_yield_data(file_path: str = 'combined_curves.csv') -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df.set_index('observation_date', inplace=True)
    df = df.dropna()

    return df


def perform_yield_curve_pca(yields_df: pd.DataFrame, n_components: int = 3):
    yield_data = yields_df.values

    scaler = StandardScaler()
    yield_data_scaled = scaler.fit_transform(yield_data)

    pca = PCA(n_components = n_components)
    principal_components = pca.fit_transform(yield_data_scaled)

    components = pca.components_

    explained_variance_ratio = pca.explained_variance_ratio_

    reconstructed_scaled = pca.inverse_transform(principal_components)

    reconstructed = scaler.inverse_transform(reconstructed_scaled)

    deviations = yield_data - reconstructed

    standard_deviations = deviations / np.std(deviations, axis=0)

    return {
        'pca_model': pca,
        'scaler': scaler,
        'principal_components': principal_components,
        'components': components,
        'explained_variance_ratio': explained_variance_ratio,
        'reconstructed': reconstructed,
        'deviations': deviations,
        'standard_deviations': standard_deviations,
        'tenors': yields_df.columns.tolist(),
    }


def plot_pca_results(pca_results, yield_df: pd.DataFrame):
    ...


def analyze_current_dislocations(pca_results, yield_df: pd.DataFrame):
    latest_dislocations = pca_results['standard_deviations'][-1]
    tenors = pca_results['tenors']

    max_cheap_idx = np.argmax(latest_dislocations)
    max_rich_idx = np.argmin(latest_dislocations)

    print(f"\nMost CHEAP: {tenors[max_cheap_idx]} ({latest_dislocations[max_cheap_idx]:+.2f})")
    print(f"Most RICH: {tenors[max_rich_idx]} ({latest_dislocations[max_rich_idx]:+.2f})")


if __name__ == '__main__':
    yield_df = load_yield_data()

    pca_results = perform_yield_curve_pca(yield_df)

    analyze_current_dislocations(pca_results, yield_df)