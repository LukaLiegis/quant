import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_yield_data(file_path: str = 'combined_curves.csv') -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df.set_index('observation_date', inplace=True)
    df = df.dropna()

    return df


def fit_pca_on_window(yield_data: np.ndarray) -> Tuple[PCA, StandardScaler]:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(yield_data)

    pca = PCA(n_components = 3)
    pca.fit(scaled_data)

    return pca, scaler


def compute_dislocation(
        yield_data: np.ndarray,
        pca_model: PCA,
        scaler: StandardScaler
) -> np.ndarray:
    scaled_current = scaler.transform(yield_data.reshape(-1, 1))

    pca_projection = pca_model.transform(scaled_current)
    reconstructed_scaled = pca_model.inverse_transform(pca_projection)
    reconstructed = scaler.inverse_transform(reconstructed_scaled)

    deviations = yield_data - reconstructed.flatten()

    standard_deviations = deviations / np.std(deviations)

    return standard_deviations


def generate_signal(
        dislocations: np.ndarray,
        tenors: list,
        dislocation_threshold: float = 2.0,
) -> Dict[str, float]:
    signals = {}

    for i, tenor in enumerate(tenors):
        dislocation = dislocations[i]

        if abs(dislocation) > dislocation_threshold:
            signals[tenor] = dislocation

    return signals


def backtest_strategy(
        yield_df: pd.DataFrame,
        window_days: int = 252
):
    start_idx = window_days

    results = []
    tenors = yield_df.columns.tolist()

    for i in range(start_idx, len(yield_df)):
        current_date = yield_df.index[i]

        window_start = i - window_days
        window_data = yield_df.iloc[window_start:i].values

        pca_model, scaler = fit_pca_on_window(window_data)

        current_yields = yield_df.iloc[i].values
        dislocations = compute_dislocation(current_yields, pca_model, scaler)

        signals = generate_signal(dislocations, tenors)

        result_row = {'date': current_date}

        for j, tenor in enumerate(tenors):
            result_row[f'{tenor}_dislocation'] = dislocations[j]
            result_row[f'{tenor}_signal'] = signals.get(tenor, 0.0)

        result_row['pca_explained_var'] = pca_model.explained_variance_ratio_.sum()

        results.append(result_row)

    return pd.DataFrame(results).set_index('date')



if __name__ == '__main__':
    yield_df = load_yield_data()

    backtest_results = backtest_strategy(yield_df)
    print(f"\nBacktest completed: {len(backtest_results)} periods analyzed")