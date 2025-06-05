import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path):
    """
    Load CRSP treasury data and prepare it for analysis
    """
    print("Loading treasury data...")

    df = pl.read_csv(
        file_path,
        schema_overrides={
            'KYCRSPID': pl.String,
            'CRSPID': pl.String,
        })
    print(f"Initial data shape: {df.shape}")

    # Show sample of data to understand format
    print("\nSample of raw data:")
    print(df.head())

    # Check data types
    print(f"\nData types:")
    print(df.dtypes)

    # Convert date columns - handle potential different formats
    try:
        df = df.with_columns([
            pl.col("CALDT").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias("CALDT_parsed")
        ])
    except:
        try:
            df = df.with_columns([
                pl.col("CALDT").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias("CALDT_parsed")
            ])
        except:
            # Try to handle as already parsed or different format
            df = df.with_columns([
                pl.col("CALDT").alias("CALDT_parsed")
            ])

    try:
        df = df.with_columns([
            pl.col("TMATDT").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias("TMATDT_parsed")
        ])
    except:
        try:
            df = df.with_columns([
                pl.col("TMATDT").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias(
                    "TMATDT_parsed")
            ])
        except:
            df = df.with_columns([
                pl.col("TMATDT").alias("TMATDT_parsed")
            ])

    print(f"After date parsing: {len(df)} rows")

    # Convert yield and duration to float, handling potential string formats
    df = df.with_columns([
        pl.col("TDYLD").cast(pl.Float64, strict=False),
        pl.col("TDDURATN").cast(pl.Float64, strict=False)
    ])

    # Check how many valid yields and durations we have
    print(f"Valid yields: {df.filter(pl.col('TDYLD').is_not_null() & (pl.col('TDYLD') > 0)).height}")
    print(f"Valid durations: {df.filter(pl.col('TDDURATN').is_not_null() & (pl.col('TDDURATN') > 0)).height}")

    # Filter out invalid yields and durations
    df = df.filter(
        (pl.col("TDYLD").is_not_null()) &
        (pl.col("TDYLD") > 0) &
        (pl.col("TDDURATN").is_not_null()) &
        (pl.col("TDDURATN") > 0) &
        (pl.col("CALDT_parsed").is_not_null()) &
        (pl.col("TMATDT_parsed").is_not_null())
    )

    print(f"After filtering for valid yields/durations: {len(df)} rows")

    if len(df) == 0:
        print("ERROR: No valid data after filtering!")
        print("Let's examine the raw data more closely...")
        df_raw = pl.read_csv(file_path)
        print("\nFirst few rows of each key column:")
        for col in ["CALDT", "TMATDT", "TDYLD", "TDDURATN"]:
            if col in df_raw.columns:
                print(f"{col}: {df_raw.select(col).head().to_pandas()[col].tolist()}")
            else:
                print(f"{col}: Column not found!")
        return df_raw  # Return raw data for debugging

    # Use the parsed date columns
    df = df.with_columns([
        pl.col("CALDT_parsed").alias("CALDT"),
        pl.col("TMATDT_parsed").alias("TMATDT")
    ]).drop(["CALDT_parsed", "TMATDT_parsed"])

    # Calculate time to maturity in years
    df = df.with_columns([
        ((pl.col("TMATDT") - pl.col("CALDT")).dt.total_days() / 365.25).alias("years_to_maturity")
    ])

    print(f"After calculating years to maturity: {len(df)} rows")

    # Filter for reasonable maturities (0.25 to 30 years) - but be more lenient initially
    df = df.filter(
        (pl.col("years_to_maturity").is_not_null()) &
        (pl.col("years_to_maturity") >= 0.01) &
        (pl.col("years_to_maturity") <= 50)  # More lenient bounds
    )

    print(f"After maturity filtering: {len(df)} rows")

    if len(df) > 0:
        print(
            f"Years to maturity range: {df.select(pl.col('years_to_maturity').min())[0, 0]:.2f} to {df.select(pl.col('years_to_maturity').max())[0, 0]:.2f}")
        print(
            f"Yield range: {df.select(pl.col('TDYLD').min())[0, 0]:.2f} to {df.select(pl.col('TDYLD').max())[0, 0]:.2f}")

    print(f"Final loaded treasury observations: {len(df)}")
    return df


def create_yield_curve_matrix(df, maturity_buckets):
    """
    Create a matrix where each row is a date and each column is a maturity bucket
    """
    print("Creating yield curve matrix...")

    # Define maturity buckets (in years)
    bucket_labels = [f"{bucket}Y" for bucket in maturity_buckets]

    # Function to assign maturity bucket
    def assign_bucket(years_to_maturity):
        bucket_idx = np.digitize(years_to_maturity, maturity_buckets) - 1
        return np.clip(bucket_idx, 0, len(maturity_buckets) - 1)

    # Add bucket assignment
    df = df.with_columns([
        pl.col("years_to_maturity").map_elements(assign_bucket, return_dtype=pl.Int32).alias("maturity_bucket")
    ])

    # For each date and bucket, take the median yield (to handle multiple bonds)
    yield_curve = df.group_by(["CALDT", "maturity_bucket"]).agg([
        pl.col("TDYLD").median().alias("yield"),
        pl.col("TDDURATN").median().alias("duration")
    ])

    # Pivot to create matrix format
    yield_matrix = yield_curve.pivot(
        index="CALDT",
        columns="maturity_bucket",
        values="yield"
    ).sort("CALDT")

    duration_matrix = yield_curve.pivot(
        index="CALDT",
        columns="maturity_bucket",
        values="duration"
    ).sort("CALDT")

    # Fill missing values with interpolation
    dates = yield_matrix.select("CALDT").to_pandas()["CALDT"].values
    yield_data = yield_matrix.drop("CALDT").to_numpy()
    duration_data = duration_matrix.drop("CALDT").to_numpy()

    # Forward fill and backward fill missing values
    for i in range(yield_data.shape[1]):
        mask = ~np.isnan(yield_data[:, i])
        if mask.any():
            yield_data[:, i] = np.interp(
                np.arange(len(yield_data)),
                np.where(mask)[0],
                yield_data[mask, i]
            )

    for i in range(duration_data.shape[1]):
        mask = ~np.isnan(duration_data[:, i])
        if mask.any():
            duration_data[:, i] = np.interp(
                np.arange(len(duration_data)),
                np.where(mask)[0],
                duration_data[mask, i]
            )

    # Remove rows with any remaining NaN values
    valid_rows = ~np.isnan(yield_data).any(axis=1) & ~np.isnan(duration_data).any(axis=1)

    return dates[valid_rows], yield_data[valid_rows], duration_data[valid_rows], maturity_buckets


def run_pca_analysis(yield_data, lookback_window=252):
    """
    Run rolling PCA analysis on yield curve data
    """
    print("Running PCA analysis...")

    n_dates, n_maturities = yield_data.shape
    n_components = min(3, n_maturities)  # Level, Slope, Curvature

    # Storage for results
    pca_results = {
        'dates': [],
        'explained_variance': [],
        'components': [],
        'reconstructed_yields': [],
        'residuals': [],
        'standardized_residuals': []
    }

    for i in range(lookback_window, n_dates):
        # Get rolling window of data
        window_data = yield_data[i - lookback_window:i]
        current_yields = yield_data[i:i + 1]

        # Standardize the data
        scaler = StandardScaler()
        scaled_window = scaler.fit_transform(window_data)
        scaled_current = scaler.transform(current_yields)

        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(scaled_window)

        # Transform current yields to PC space and back
        pc_scores = pca.transform(scaled_current)
        reconstructed_scaled = pca.inverse_transform(pc_scores)
        reconstructed_yields = scaler.inverse_transform(reconstructed_scaled)[0]

        # Calculate residuals
        residuals = current_yields[0] - reconstructed_yields

        # Standardize residuals using historical volatility
        window_residuals = []
        for j in range(max(0, i - lookback_window), i - 1):
            if j >= lookback_window:
                past_window = yield_data[j - lookback_window:j]
                past_scaled = scaler.fit_transform(past_window)
                past_pca = PCA(n_components=n_components)
                past_pca.fit(past_scaled)

                past_current = yield_data[j:j + 1]
                past_scaled_current = scaler.transform(past_current)
                past_pc_scores = past_pca.transform(past_scaled_current)
                past_reconstructed_scaled = past_pca.inverse_transform(past_pc_scores)
                past_reconstructed = scaler.inverse_transform(past_reconstructed_scaled)[0]

                window_residuals.append(past_current[0] - past_reconstructed)

        if len(window_residuals) > 20:  # Need minimum history
            window_residuals = np.array(window_residuals)
            residual_std = np.std(window_residuals, axis=0)
            residual_std = np.where(residual_std > 0, residual_std, 1)  # Avoid division by zero
            standardized_residuals = residuals / residual_std
        else:
            standardized_residuals = np.zeros_like(residuals)

        # Store results
        pca_results['dates'].append(i)
        pca_results['explained_variance'].append(pca.explained_variance_ratio_)
        pca_results['components'].append(pca.components_)
        pca_results['reconstructed_yields'].append(reconstructed_yields)
        pca_results['residuals'].append(residuals)
        pca_results['standardized_residuals'].append(standardized_residuals)

    return pca_results


def generate_trading_signals(pca_results, duration_data, maturity_buckets, threshold=2.0):
    """
    Generate trading signals based on PCA dislocations
    """
    print("Generating trading signals...")

    signals = []

    for i, standardized_residuals in enumerate(pca_results['standardized_residuals']):
        date_idx = pca_results['dates'][i]
        current_durations = duration_data[date_idx]

        # Find significant dislocations (absolute z-score > threshold)
        significant_mask = np.abs(standardized_residuals) > threshold

        if not significant_mask.any():
            # No significant dislocations
            signals.append({
                'date_idx': date_idx,
                'positions': np.zeros(len(maturity_buckets)),
                'max_dislocation': 0
            })
            continue

        # Create mean-reverting positions
        # Negative residual (rich) -> sell (negative position)
        # Positive residual (cheap) -> buy (positive position)
        raw_positions = -standardized_residuals  # Flip sign for mean reversion

        # Only trade where dislocations are significant
        positions = np.where(significant_mask, raw_positions, 0)

        # Scale positions by duration (risk management)
        # Normalize by duration to create duration-neutral positions
        avg_duration = np.mean(current_durations)
        duration_weights = avg_duration / current_durations
        positions = positions * duration_weights

        # Scale total exposure
        total_exposure = np.sum(np.abs(positions))
        if total_exposure > 0:
            positions = positions / total_exposure  # Normalize to unit exposure

        signals.append({
            'date_idx': date_idx,
            'positions': positions,
            'max_dislocation': np.max(np.abs(standardized_residuals))
        })

    return signals


def backtest_strategy(signals, yield_data, dates, maturity_buckets, transaction_cost=0.001):
    """
    Backtest the PCA trading strategy
    """
    print("Running backtest...")

    n_assets = len(maturity_buckets)
    portfolio_value = 1000000  # Start with $1M
    positions = np.zeros(n_assets)

    backtest_results = {
        'dates': [],
        'portfolio_values': [],
        'daily_returns': [],
        'positions': [],
        'trade_costs': []
    }

    for i, signal in enumerate(signals):
        if i == 0:
            backtest_results['dates'].append(dates[signal['date_idx']])
            backtest_results['portfolio_values'].append(portfolio_value)
            backtest_results['daily_returns'].append(0)
            backtest_results['positions'].append(positions.copy())
            backtest_results['trade_costs'].append(0)
            continue

        prev_date_idx = signals[i - 1]['date_idx']
        curr_date_idx = signal['date_idx']

        if curr_date_idx >= len(yield_data):
            break

        # Calculate P&L from yield changes
        yield_change = yield_data[curr_date_idx] - yield_data[prev_date_idx]

        # P&L is approximately -duration * position_size * yield_change * portfolio_value
        # Simplified assumption: each position represents 1% of portfolio per unit
        daily_pnl = -np.sum(positions * yield_change) * portfolio_value * 0.01

        # Calculate new positions and transaction costs
        new_positions = signal['positions']
        position_change = np.abs(new_positions - positions)
        trade_cost = np.sum(position_change) * portfolio_value * transaction_cost

        # Update portfolio
        portfolio_value += daily_pnl - trade_cost
        daily_return = (daily_pnl - trade_cost) / backtest_results['portfolio_values'][-1]

        positions = new_positions.copy()

        # Store results
        backtest_results['dates'].append(dates[curr_date_idx])
        backtest_results['portfolio_values'].append(portfolio_value)
        backtest_results['daily_returns'].append(daily_return)
        backtest_results['positions'].append(positions.copy())
        backtest_results['trade_costs'].append(trade_cost)

    return backtest_results


def analyze_performance(backtest_results):
    """
    Analyze strategy performance
    """
    returns = np.array(backtest_results['daily_returns'][1:])  # Skip first zero return
    portfolio_values = backtest_results['portfolio_values']

    # Performance metrics
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1

    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns)

    # Win rate
    win_rate = np.mean(returns > 0)

    print("\n=== STRATEGY PERFORMANCE ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")

    # Calculate average trade costs
    total_trade_costs = sum(backtest_results['trade_costs'])
    print(f"Total Transaction Costs: ${total_trade_costs:,.2f}")

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }


def plot_results(backtest_results, pca_results, maturity_buckets):
    """
    Plot strategy results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Portfolio value over time
    axes[0, 0].plot(backtest_results['dates'], backtest_results['portfolio_values'])
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True)

    # Daily returns distribution
    returns = backtest_results['daily_returns'][1:]
    axes[0, 1].hist(returns, bins=50, alpha=0.7)
    axes[0, 1].set_title('Daily Returns Distribution')
    axes[0, 1].set_xlabel('Daily Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True)

    # Average PCA components (Level, Slope, Curvature)
    if pca_results['components']:
        avg_components = np.mean(pca_results['components'], axis=0)
        for i in range(min(3, avg_components.shape[0])):
            axes[1, 0].plot(maturity_buckets, avg_components[i],
                            label=f'PC{i + 1} ({"Level" if i == 0 else "Slope" if i == 1 else "Curvature"})')
        axes[1, 0].set_title('Average PCA Components')
        axes[1, 0].set_xlabel('Maturity (Years)')
        axes[1, 0].set_ylabel('Component Loading')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Explained variance over time
    if pca_results['explained_variance']:
        explained_var = np.array(pca_results['explained_variance'])
        cumulative_var = np.cumsum(explained_var, axis=1)

        axes[1, 1].plot(np.mean(cumulative_var, axis=0), 'o-')
        axes[1, 1].set_title('Average Cumulative Explained Variance')
        axes[1, 1].set_xlabel('Principal Component')
        axes[1, 1].set_ylabel('Cumulative Explained Variance')
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the PCA yield curve trading strategy
    """
    # Parameters
    file_path = "data/crsp_a_treasuries.csv.gz"  # Update with your file path
    maturity_buckets = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])  # Maturity buckets in years
    lookback_window = 252  # 1 year of trading days for PCA estimation
    signal_threshold = 2.0  # Z-score threshold for trading signals
    transaction_cost = 0.001  # 10bps transaction cost

    # Load and prepare data
    df = load_and_prepare_data(file_path)

    # Create yield curve matrix
    dates, yield_data, duration_data, maturity_buckets = create_yield_curve_matrix(df, maturity_buckets)

    print(f"Created yield curve matrix: {yield_data.shape[0]} dates x {yield_data.shape[1]} maturities")

    # Run PCA analysis
    pca_results = run_pca_analysis(yield_data, lookback_window)

    # Generate trading signals
    signals = generate_trading_signals(pca_results, duration_data, maturity_buckets, signal_threshold)

    # Backtest strategy
    backtest_results = backtest_strategy(signals, yield_data, dates, maturity_buckets, transaction_cost)

    # Analyze performance
    performance_metrics = analyze_performance(backtest_results)

    # Plot results
    plot_results(backtest_results, pca_results, maturity_buckets)

    return backtest_results, pca_results, performance_metrics


if __name__ == "__main__":
    results = main()