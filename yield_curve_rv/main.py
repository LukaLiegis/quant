import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_and_prepare_data():
    df = pd.read_csv('combined_curves.csv')
    df['observation_date'] = pd.to_datetime(df['observation_date'])

    yield_columns = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3',
                     'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']

    yields = df[yield_columns].apply(pd.to_numeric, errors='coerce')

    maturity_labels = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
    maturity_years = [1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]

    return df, yields, yield_columns, maturity_labels, maturity_years


def compute_yield_changes(
        yields: pd.DataFrame,
):
    yield_changes = yields.diff().dropna()
    yield_changes = yield_changes.dropna()

    return yield_changes


def validate_data(
        yields,
        maturity_labels,
):
    missing_pct = yields.isnull().sum() / len(yields) * 100
    for label, pct in zip(maturity_labels, missing_pct):
        if pct > 0:
            print(f"{label}: {pct:.1f}% missing")


def perform_pca_analysis(
        yield_changes
):
    pca = PCA()
    pca_result = pca.fit_transform(yield_changes)

    components = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance = pca.explained_variance_

    pc_std_devs = np.sqrt(explained_variance)

    return pca, components, explained_variance_ratio, explained_variance, pc_std_devs, pca_result


def print_pca_summary(
        explained_variance_ratio,
        pc_std_devs
):

    print(f"{'Component':<12} {'Std Dev (bp)':<15} {'Proportion':<12} {'Cumulative':<12}")
    print("-" * 60)

    cumulative = 0
    for i in range(min(5, len(explained_variance_ratio))):
        cumulative += explained_variance_ratio[i]
        print(f"PC #{i + 1:<8} {pc_std_devs[i] * 100:<15.2f} {explained_variance_ratio[i]:<12.1%} {cumulative:<12.1%}")

    print(f"\nFirst three components explain: {explained_variance_ratio[:3].sum():.1%} of total variance")


def plot_principal_components(
        components,
        explained_variance_ratio,
        maturity_years,
        maturity_labels
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    component_names = ['Level', 'Slope', 'Curvature']
    colors = ['blue', 'red', 'green']

    for i in range(3):
        axes[i].plot(maturity_years, components[i] * 100, 'o-',
                     linewidth=2, markersize=6, color=colors[i], label=f'PC #{i + 1}')
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[i].set_title(f'PC #{i + 1}: {component_names[i]} ({explained_variance_ratio[i]:.1%} of variance)')
        axes[i].set_xlabel('Years to Maturity')
        axes[i].set_ylabel('Yield Change (bp)')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(maturity_years)
        axes[i].set_xticklabels(maturity_labels)

        if i == 0:
            axes[i].text(0.02, 0.95, 'Positive at all maturities\n(Level shift)',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        elif i == 1:
            axes[i].text(0.02, 0.95, 'Negative at short end,\nPositive at long end\n(Slope change)',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        else:
            axes[i].text(0.02, 0.95, 'Positive at ends,\nNegative in middle\n(Curvature change)',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.suptitle('Principal Components of Yield Curve Changes\n(Similar to Figure 2 in Solomon Brothers Paper)',
                 fontsize=14, fontweight='bold')
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
    df, yields, yield_columns, maturity_labels, maturity_years = load_and_prepare_data()

    yield_changes = compute_yield_changes(yields)

    pca, components, explained_variance_ratio, explained_variance, pc_std_devs, pca_result = perform_pca_analysis(
        yield_changes)

    print_pca_summary(explained_variance_ratio, pc_std_devs)

    corr_matrix = analyze_yield_changes_statistics(yield_changes, maturity_labels)

    plot_principal_components(components, explained_variance_ratio, maturity_years, maturity_labels)
    plot_variance_explained(explained_variance_ratio)

    return {
        'pca': pca,
        'components': components,
        'explained_variance_ratio': explained_variance_ratio,
        'yield_changes': yield_changes,
        'maturity_years': maturity_years,
        'maturity_labels': maturity_labels,
        'pca_result': pca_result
    }


if __name__ == "__main__":
    results = main()