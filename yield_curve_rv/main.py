import numpy as np
import polars as pl
from typing import Tuple, Dict
from matplotlib import pyplot as plt


def load_yield_data() -> pl.DataFrame:
    df = pl.read_csv("combined_curves.csv")
    df = df.with_columns(
        pl.col('observation_date').str.to_date()
    ).sort("observation_date")
    df = df.drop_nulls()
    return df


def calculate_yield_changes(
        df: pl.DataFrame
) -> pl.DataFrame:
    yield_columns = [col for col in df.columns if col.startswith("DGS")]

    changes_df = df.select([
        pl.col('observation_date'),
        *[pl.col(col).diff().alias(f'{col}_change') for col in yield_columns]
    ]).drop_nulls()

    return changes_df


def perform_pca(
        changes_df: pl.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    change_columns = [col for col in changes_df.columns if col.endswith("_change")]

    yield_changes_matrix = changes_df.select(change_columns).to_numpy()

    mask = ~np.isnan(yield_changes_matrix).any(axis=1)
    yield_changes_matrix = yield_changes_matrix[mask]

    print(f'Data shape: {yield_changes_matrix.shape}')
    print(f'Number of observations: {yield_changes_matrix.shape[0]}')
    print(f'Number of point: {yield_changes_matrix.shape[1]}')

    cov_matrix = np.cov(yield_changes_matrix.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, yield_changes_matrix


def analyze_components(
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
) -> Dict:
    total_variance = np.sum(eigenvalues)
    variance_explained = eigenvalues / total_variance * 100
    cumulative_variance = np.cumsum(variance_explained)

    std_devs = np.sqrt(eigenvalues)

    analysis = {
        'eigenvalues': eigenvalues,
        'std_devs': std_devs,
        'variance_explained': variance_explained,
        'cumulative_variance': cumulative_variance,
        'eigenvectors': eigenvectors,
    }

    for i in range(min(3, len(eigenvalues))):
        print(f"Component {i + 1}:")
        print(f"  Standard Deviation: {std_devs[i]:.2f} bp")
        print(f"  Variance Explained: {variance_explained[i]:.1f}%")
        print(f"  Cumulative Variance: {cumulative_variance[i]:.1f}%")
        print()

    return analysis


def get_maturity_mapping() -> Dict[str, float]:
    return {
        'DGS1MO_change': 1/12,
        'DGS3MO_change': 0.25,
        'DGS6MO_change': 0.5,
        'DGS1_change': 1.0,
        'DGS2_change': 2.0,
        'DGS3_change': 3.0,
        'DGS5_change': 5.0,
        'DGS7_change': 7.0,
        'DGS10_change': 10.0,
        'DGS20_change': 20.0,
        'DGS30_change': 30.0
    }


def plot_principal_components(
        analysis: Dict,
        changes_df: pl.DataFrame,
) -> None:
    maturity_map = get_maturity_mapping()
    change_columns = [col for col in changes_df.columns if col.endswith('_change')]
    maturities = [maturity_map[col] for col in change_columns]

    eigenvectors_scaled = analysis['eigenvectors'] * analysis['std_devs'].reshape(1, -1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    components = [
        {'name': 'PC #1: Level', 'color': 'blue', 'interpretation': 'Level Shift'},
        {'name': 'PC #2: Slope', 'color': 'red', 'interpretation': 'Slope Change'},
        {'name': 'PC #3: Curvature', 'color': 'green', 'interpretation': 'Curvature Change'}
    ]

    for i, (ax, comp) in enumerate(zip(axes, components)):
        ax.plot(maturities, eigenvectors_scaled[:, i], 'o-',
                color=comp['color'], linewidth=2, markersize=6, label=comp['name'])

        ax.set_xlabel('Years to Maturity')
        ax.set_ylabel('Yield Change (bp)')
        ax.set_title(f"{comp['name']} - {comp['interpretation']} "
                     f"({analysis['variance_explained'][i]:.1f}% of variance)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        y_max = max(abs(eigenvectors_scaled[:, i].min()), abs(eigenvectors_scaled[:, i].max()))
        ax.set_ylim(-y_max * 1.1, y_max * 1.1)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_variance_explained(
        analysis: Dict,
) -> None:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_components = min(10, len(analysis['variance_explained']))
    ax1.bar(range(1, n_components + 1), analysis['variance_explained'][:n_components])
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Variance Explained by Each Component')
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, n_components + 1), analysis['cumulative_variance'][:n_components], 'o-')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Variance Explained')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=99, color='red', linestyle='--', alpha=0.7, label='99%')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def interpret_components(
        analysis: Dict,
        changes_df: pl.DataFrame
):
    change_columns = [col for col in changes_df.columns if col.endswith('_change')]
    maturity_map = get_maturity_mapping()

    pc1 = analysis['eigenvectors'][:, 0]
    print("First Component (Level Shift):")
    print(f"  All coefficients positive: {np.all(pc1 > 0)}")
    print(f"  Variance explained: {analysis['variance_explained'][0]:.1f}%")
    print(f"  Standard deviation: {analysis['std_devs'][0]:.2f} bp")
    print()

    pc2 = analysis['eigenvectors'][:, 1]
    short_end_negative = pc2[0] < 0
    long_end_positive = pc2[-1] > 0
    print("Second Component (Slope Change):")
    print(f"  Short end negative: {short_end_negative}")
    print(f"  Long end positive: {long_end_positive}")
    print(f"  Variance explained: {analysis['variance_explained'][1]:.1f}%")
    print(f"  Standard deviation: {analysis['std_devs'][1]:.2f} bp")
    print()

    pc3 = analysis['eigenvectors'][:, 2]
    print("Third Component (Curvature Change):")
    print(f"  Variance explained: {analysis['variance_explained'][2]:.1f}%")
    print(f"  Standard deviation: {analysis['std_devs'][2]:.2f} bp")
    print()

    print(f"Total variance explained by first 3 components: {analysis['cumulative_variance'][2]:.1f}%")


def main():
    df = load_yield_data()
    changes_df = calculate_yield_changes(df)
    eigenvalues, eigenvectors, changes_matrix = perform_pca(changes_df)
    analysis = analyze_components(eigenvalues, eigenvectors)
    interpret_components(analysis, changes_df)

    plot_principal_components(analysis, changes_df)
    plot_variance_explained(analysis)


if __name__ == "__main__":
    main()