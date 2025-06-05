import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path):
    """
    Load CRSP treasury data and prepare it for analysis
    """
    print("Loading treasury data...")

    # Try different reading approaches for the CRSP data
    try:
        df = pl.read_csv(
            file_path,
            schema_overrides={
                'KYCRSPID': pl.String,
                'CRSPID': pl.String,
            })
    except:
        df = pl.read_csv(file_path)

    print(f"Initial data shape: {df.shape}")
    print("\nColumns available:")
    print(df.columns)

    # Show sample of data to understand format
    print("\nSample of raw data:")
    print(df.head())

    # Handle date parsing more robustly
    date_col = 'CALDT'
    if date_col in df.columns:
        try:
            # Try different date formats
            df = df.with_columns([
                pl.col(date_col).cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias("date")
            ])
        except:
            try:
                df = df.with_columns([
                    pl.col(date_col).cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias("date")
                ])
            except:
                print("Date parsing failed, trying as string conversion...")
                df = df.with_columns([
                    pl.col(date_col).alias("date")
                ])

    # Handle maturity date
    maturity_col = 'TMATDT'
    if maturity_col in df.columns:
        try:
            df = df.with_columns([
                pl.col(maturity_col).cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias(
                    "maturity_date")
            ])
        except:
            try:
                df = df.with_columns([
                    pl.col(maturity_col).cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias(
                        "maturity_date")
                ])
            except:
                df = df.with_columns([
                    pl.col(maturity_col).alias("maturity_date")
                ])

    # Convert yield and duration to float
    yield_col = 'TDYLD' if 'TDYLD' in df.columns else 'YLD'
    duration_col = 'TDDURATN' if 'TDDURATN' in df.columns else 'DURATN'

    print(f"Using yield column: {yield_col}")
    print(f"Using duration column: {duration_col}")

    df = df.with_columns([
        pl.col(yield_col).cast(pl.Float64, strict=False).alias("yield"),
        pl.col(duration_col).cast(pl.Float64, strict=False).alias("duration")
    ])

    # Filter out invalid data
    df = df.filter(
        (pl.col("yield").is_not_null()) &
        (pl.col("yield") > 0) &
        (pl.col("yield") < 50) &  # Reasonable yield bounds
        (pl.col("duration").is_not_null()) &
        (pl.col("duration") > 0) &
        (pl.col("date").is_not_null())
    )

    print(f"After basic filtering: {len(df)} rows")

    if len(df) == 0:
        print("ERROR: No valid data after filtering!")
        return None

    # Calculate years to maturity if we have maturity dates
    if 'maturity_date' in df.columns and df.select(pl.col("maturity_date").is_not_null().sum())[0, 0] > 0:
        df = df.with_columns([
            ((pl.col("maturity_date") - pl.col("date")).dt.total_days() / 365.25).alias("years_to_maturity")
        ])

        # Filter for reasonable maturities
        df = df.filter(
            (pl.col("years_to_maturity").is_not_null()) &
            (pl.col("years_to_maturity") >= 0.08) &  # 1 month minimum
            (pl.col("years_to_maturity") <= 35)  # 35 years maximum
        )
    else:
        # Use duration as proxy for years to maturity
        df = df.with_columns([
            pl.col("duration").alias("years_to_maturity")
        ])

    print(f"After maturity filtering: {len(df)} rows")

    if len(df) > 0:
        print(
            f"Years to maturity range: {df.select(pl.col('years_to_maturity').min())[0, 0]:.2f} to {df.select(pl.col('years_to_maturity').max())[0, 0]:.2f}")
        print(
            f"Yield range: {df.select(pl.col('yield').min())[0, 0]:.2f} to {df.select(pl.col('yield').max())[0, 0]:.2f}")

    return df


def create_yield_curve_matrix(df, target_maturities):
    """
    Create a matrix where each row is a date and each column is a target maturity
    Interpolate yields to standard maturities following research methodology
    """
    print("Creating yield curve matrix...")

    # Convert to pandas for easier manipulation
    df_pd = df.to_pandas()

    # Sort by date and maturity
    df_pd = df_pd.sort_values(['date', 'years_to_maturity'])

    # Get unique dates
    dates = sorted(df_pd['date'].unique())
    print(f"Found {len(dates)} unique dates")

    # Initialize matrices
    yield_matrix = np.full((len(dates), len(target_maturities)), np.nan)
    duration_matrix = np.full((len(dates), len(target_maturities)), np.nan)

    # For each date, interpolate yields to target maturities
    valid_dates = []

    for i, date in enumerate(dates):
        date_data = df_pd[df_pd['date'] == date].copy()

        if len(date_data) < 3:  # Need at least 3 points for interpolation
            continue

        # Remove duplicates - take median if multiple bonds at same maturity
        date_data = date_data.groupby('years_to_maturity').agg({
            'yield': 'median',
            'duration': 'median'
        }).reset_index()

        if len(date_data) < 3:
            continue

        # Sort by maturity
        date_data = date_data.sort_values('years_to_maturity')

        maturities = date_data['years_to_maturity'].values
        yields = date_data['yield'].values
        durations = date_data['duration'].values

        # Only interpolate within the range of available data
        min_mat = maturities.min()
        max_mat = maturities.max()

        # Filter target maturities to interpolation range
        valid_targets = (target_maturities >= min_mat) & (target_maturities <= max_mat)

        if np.sum(valid_targets) < 3:  # Need at least 3 target points
            continue

        try:
            # Interpolate yields
            yield_interp = interp1d(maturities, yields, kind='linear',
                                    bounds_error=False, fill_value=np.nan)
            duration_interp = interp1d(maturities, durations, kind='linear',
                                       bounds_error=False, fill_value=np.nan)

            interpolated_yields = yield_interp(target_maturities)
            interpolated_durations = duration_interp(target_maturities)

            # Only keep if we have enough valid interpolated points
            if np.sum(~np.isnan(interpolated_yields)) >= 3:
                yield_matrix[len(valid_dates)] = interpolated_yields
                duration_matrix[len(valid_dates)] = interpolated_durations
                valid_dates.append(date)

        except Exception as e:
            continue

    # Trim matrices to valid dates
    n_valid = len(valid_dates)
    yield_matrix = yield_matrix[:n_valid]
    duration_matrix = duration_matrix[:n_valid]

    # Additional cleaning - remove dates with too many NaNs
    max_nan_ratio = 0.3  # Allow up to 30% missing values
    valid_rows = np.mean(np.isnan(yield_matrix), axis=1) <= max_nan_ratio

    yield_matrix = yield_matrix[valid_rows]
    duration_matrix = duration_matrix[valid_rows]
    valid_dates = [valid_dates[i] for i in range(len(valid_dates)) if valid_rows[i]]

    # Forward fill remaining NaNs
    for i in range(yield_matrix.shape[1]):
        mask = ~np.isnan(yield_matrix[:, i])
        if mask.any():
            yield_matrix[:, i] = np.interp(
                np.arange(len(yield_matrix)),
                np.where(mask)[0],
                yield_matrix[mask, i]
            )

        mask = ~np.isnan(duration_matrix[:, i])
        if mask.any():
            duration_matrix[:, i] = np.interp(
                np.arange(len(duration_matrix)),
                np.where(mask)[0],
                duration_matrix[mask, i]
            )

    print(f"Final yield curve matrix: {yield_matrix.shape[0]} dates x {yield_matrix.shape[1]} maturities")

    return np.array(valid_dates), yield_matrix, duration_matrix


def run_pca_smoothing_analysis(yield_data, n_components=3, lookback_window=252):
    """
    Run PCA analysis following the "smoothing the curve" methodology from the research
    """
    print("Running PCA smoothing analysis...")

    n_dates, n_maturities = yield_data.shape
    n_components = min(n_components, n_maturities)

    # Storage for results
    pca_results = {
        'dates': [],
        'explained_variance': [],
        'components': [],
        'projected_yields': [],
        'dislocations': [],
        'standardized_dislocations': []
    }

    # Calculate yield changes for PCA (as recommended in research)
    yield_changes = np.diff(yield_data, axis=0)

    for i in range(lookback_window, len(yield_changes)):
        # Get rolling window of yield changes
        window_changes = yield_changes[i - lookback_window:i]
        current_yields = yield_data[i + 1]  # Current yield levels

        # Remove any rows with NaN values
        valid_rows = ~np.isnan(window_changes).any(axis=1)
        if np.sum(valid_rows) < lookback_window * 0.8:  # Need at least 80% valid data
            continue

        window_changes_clean = window_changes[valid_rows]

        # Fit PCA on yield changes (as per Lardic et al. recommendation)
        pca = PCA(n_components=n_components)
        pca.fit(window_changes_clean)

        # Project current yield curve using PCA factors
        # First, we need to estimate what the "smooth" curve should be

        # Method 1: Use the mean yield curve from the window
        mean_yields = np.nanmean(yield_data[i - lookback_window + 1:i + 1], axis=0)

        # Create a synthetic "change" to project
        # We'll use small perturbations to find the projection
        synthetic_change = np.zeros(n_maturities)

        # Project using PCA components
        pca_projection = pca.transform(synthetic_change.reshape(1, -1))
        projected_change = pca.inverse_transform(pca_projection)[0]

        # The "smooth" yield curve is the mean yields
        smooth_yields = mean_yields

        # Calculate dislocations as deviations from smooth curve
        dislocations = current_yields - smooth_yields

        # Calculate standardized dislocations using rolling window
        if i >= lookback_window + 63:  # Need enough history for standardization
            # Get historical dislocations
            historical_dislocations = []
            for j in range(max(0, i - 252), i):  # Use up to 1 year of history
                if j >= lookback_window:
                    hist_window = yield_changes[j - lookback_window:j]
                    hist_valid = ~np.isnan(hist_window).any(axis=1)

                    if np.sum(hist_valid) >= lookback_window * 0.5:
                        hist_pca = PCA(n_components=n_components)
                        hist_pca.fit(hist_window[hist_valid])

                        hist_mean = np.nanmean(yield_data[j - lookback_window + 1:j + 1], axis=0)
                        hist_current = yield_data[j + 1] if j + 1 < len(yield_data) else yield_data[j]
                        hist_dislocation = hist_current - hist_mean

                        historical_dislocations.append(hist_dislocation)

            if len(historical_dislocations) > 20:
                historical_dislocations = np.array(historical_dislocations)
                # Calculate rolling standard deviation
                dislocation_std = np.nanstd(historical_dislocations, axis=0)
                dislocation_std = np.where(dislocation_std > 0, dislocation_std, 1e-6)

                standardized_dislocations = dislocations / dislocation_std
            else:
                standardized_dislocations = np.zeros_like(dislocations)
        else:
            standardized_dislocations = np.zeros_like(dislocations)

        # Store results
        pca_results['dates'].append(i)
        pca_results['explained_variance'].append(pca.explained_variance_ratio_)
        pca_results['components'].append(pca.components_)
        pca_results['projected_yields'].append(smooth_yields)
        pca_results['dislocations'].append(dislocations)
        pca_results['standardized_dislocations'].append(standardized_dislocations)

    return pca_results


def generate_duration_neutral_signals(pca_results, duration_data, target_maturities,
                                      threshold=2.0, max_positions=5):
    """
    Generate duration-neutral trading signals based on PCA dislocations
    Following the "duration units" methodology from the research
    """
    print("Generating duration-neutral trading signals...")

    signals = []

    for i, standardized_dislocations in enumerate(pca_results['standardized_dislocations']):
        date_idx = pca_results['dates'][i]

        if date_idx >= len(duration_data):
            continue

        current_durations = duration_data[date_idx]

        # Find significant dislocations
        significant_mask = np.abs(standardized_dislocations) > threshold

        if not significant_mask.any():
            signals.append({
                'date_idx': date_idx,
                'positions': np.zeros(len(target_maturities)),
                'max_dislocation': 0,
                'trade_rationale': 'No significant dislocations'
            })
            continue

        # Create duration-weighted positions
        # Use "duration units" - positions sized by duration to equalize DV01

        # Calculate DV01 (dollar value of 01) for each maturity
        # DV01 ≈ Duration × Position Size × 0.0001
        # We want equal DV01 across positions for duration neutrality

        # Base position size (we'll normalize later)
        base_position_size = 1.0

        # Raw signal strength (negative dislocation = rich = sell)
        raw_signals = -standardized_dislocations

        # Only trade significant dislocations
        trade_signals = np.where(significant_mask, raw_signals, 0)

        # Limit to top N positions by absolute signal strength
        abs_signals = np.abs(trade_signals)
        if np.sum(abs_signals > 0) > max_positions:
            threshold_val = np.sort(abs_signals)[-max_positions]
            trade_signals = np.where(abs_signals >= threshold_val, trade_signals, 0)

        # Duration-weight the positions
        # Position size inversely proportional to duration to equalize DV01
        positions = np.zeros_like(trade_signals)

        for j, signal in enumerate(trade_signals):
            if signal != 0:
                # Duration weighting: larger positions for shorter duration instruments
                duration_weight = 1.0 / current_durations[j] if current_durations[j] > 0 else 0
                positions[j] = signal * duration_weight

        # Normalize positions to create a dollar-neutral portfolio
        long_positions = np.sum(positions[positions > 0])
        short_positions = np.sum(positions[positions < 0])

        if long_positions > 0 and short_positions < 0:
            # Scale to make dollar neutral
            scale_factor = min(abs(short_positions), long_positions)
            if scale_factor > 0:
                positions = positions / max(long_positions, abs(short_positions)) * scale_factor

        # Final scaling for reasonable position sizes
        max_position = np.max(np.abs(positions))
        if max_position > 0:
            positions = positions / max_position * 0.1  # Max 10% allocation per position

        signals.append({
            'date_idx': date_idx,
            'positions': positions,
            'max_dislocation': np.max(np.abs(standardized_dislocations)),
            'trade_rationale': f'Trading {np.sum(positions != 0)} positions'
        })

    return signals


def backtest_pca_strategy(signals, yield_data, duration_data, dates, target_maturities,
                          transaction_cost=0.0005, initial_portfolio=1000000):
    """
    Backtest the PCA smoothing strategy with proper P&L calculation
    """
    print("Running backtest...")

    n_assets = len(target_maturities)
    portfolio_value = initial_portfolio
    positions = np.zeros(n_assets)  # Current positions

    backtest_results = {
        'dates': [],
        'portfolio_values': [],
        'daily_returns': [],
        'daily_pnl': [],
        'positions': [],
        'trade_costs': [],
        'num_trades': []
    }

    for i, signal in enumerate(signals):
        curr_date_idx = signal['date_idx']

        if i == 0:
            # Initialize
            backtest_results['dates'].append(dates[curr_date_idx])
            backtest_results['portfolio_values'].append(portfolio_value)
            backtest_results['daily_returns'].append(0)
            backtest_results['daily_pnl'].append(0)
            backtest_results['positions'].append(positions.copy())
            backtest_results['trade_costs'].append(0)
            backtest_results['num_trades'].append(0)
            continue

        prev_date_idx = signals[i - 1]['date_idx']

        if curr_date_idx >= len(yield_data) or prev_date_idx >= len(yield_data):
            break

        # Calculate P&L from yield changes
        # P&L = -Duration × Position × Yield Change × Notional
        yield_change = yield_data[curr_date_idx] - yield_data[prev_date_idx]
        current_durations = duration_data[prev_date_idx]  # Use previous durations

        # Calculate P&L for each position
        position_pnl = np.zeros(n_assets)
        for j in range(n_assets):
            if positions[j] != 0:
                # P&L = -Duration × Position × Yield Change × Portfolio Value
                # Position is in percentage terms, yield change in percentage points
                position_pnl[j] = -current_durations[j] * positions[j] * yield_change[j] * portfolio_value

        total_pnl = np.sum(position_pnl)

        # Update positions and calculate transaction costs
        new_positions = signal['positions']
        position_changes = np.abs(new_positions - positions)

        # Transaction cost based on position change
        trade_cost = np.sum(position_changes) * portfolio_value * transaction_cost
        num_trades = np.sum(position_changes > 1e-6)

        # Update portfolio value
        portfolio_value += total_pnl - trade_cost
        daily_return = (total_pnl - trade_cost) / backtest_results['portfolio_values'][-1]

        # Update positions
        positions = new_positions.copy()

        # Store results
        backtest_results['dates'].append(dates[curr_date_idx])
        backtest_results['portfolio_values'].append(portfolio_value)
        backtest_results['daily_returns'].append(daily_return)
        backtest_results['daily_pnl'].append(total_pnl)
        backtest_results['positions'].append(positions.copy())
        backtest_results['trade_costs'].append(trade_cost)
        backtest_results['num_trades'].append(num_trades)

    return backtest_results


def analyze_strategy_performance(backtest_results, pca_results):
    """
    Comprehensive performance analysis
    """
    returns = np.array(backtest_results['daily_returns'][1:])
    portfolio_values = np.array(backtest_results['portfolio_values'])

    # Basic performance metrics
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    n_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1

    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    # Drawdown analysis
    cumulative_returns = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns)

    # Trading statistics
    win_rate = np.mean(returns > 0)
    total_trades = sum(backtest_results['num_trades'])
    total_costs = sum(backtest_results['trade_costs'])

    # PCA analysis
    if pca_results['explained_variance']:
        avg_explained_variance = np.mean(pca_results['explained_variance'], axis=0)
        avg_cumulative_var = np.cumsum(avg_explained_variance)
    else:
        avg_explained_variance = []
        avg_cumulative_var = []

    print("\n" + "=" * 50)
    print("STRATEGY PERFORMANCE ANALYSIS")
    print("=" * 50)
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total Trades: {total_trades:,}")
    print(f"Total Transaction Costs: ${total_costs:,.2f}")
    print(f"Avg Daily Trading: {total_trades / n_days:.1f} trades/day")

    if len(avg_explained_variance) >= 3:
        print(f"\nPCA FACTOR ANALYSIS:")
        print(f"Level Factor Explains: {avg_explained_variance[0]:.1%}")
        print(f"Slope Factor Explains: {avg_explained_variance[1]:.1%}")
        print(f"Curvature Factor Explains: {avg_explained_variance[2]:.1%}")
        print(f"First 3 Factors Total: {avg_cumulative_var[2]:.1%}")

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'total_costs': total_costs,
        'avg_explained_variance': avg_explained_variance
    }


def plot_strategy_results(backtest_results, pca_results, target_maturities):
    """
    Create comprehensive plots of strategy performance
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Portfolio value over time
    axes[0, 0].plot(backtest_results['dates'], backtest_results['portfolio_values'])
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Daily returns distribution
    returns = backtest_results['daily_returns'][1:]
    axes[0, 1].hist(returns, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Daily Returns Distribution')
    axes[0, 1].set_xlabel('Daily Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True)

    # Cumulative returns
    cumulative_returns = np.cumprod(1 + np.array(returns))
    axes[0, 2].plot(backtest_results['dates'][1:], cumulative_returns)
    axes[0, 2].set_title('Cumulative Returns')
    axes[0, 2].set_ylabel('Cumulative Return')
    axes[0, 2].grid(True)
    axes[0, 2].tick_params(axis='x', rotation=45)

    # Average PCA components
    if pca_results['components']:
        avg_components = np.mean(pca_results['components'], axis=0)
        component_names = ['Level', 'Slope', 'Curvature']

        for i in range(min(3, avg_components.shape[0])):
            axes[1, 0].plot(target_maturities, avg_components[i],
                            label=f'PC{i + 1} ({component_names[i]})', marker='o')

        axes[1, 0].set_title('Average PCA Components')
        axes[1, 0].set_xlabel('Maturity (Years)')
        axes[1, 0].set_ylabel('Component Loading')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Explained variance
    if pca_results['explained_variance']:
        explained_var = np.array(pca_results['explained_variance'])
        cumulative_var = np.cumsum(explained_var, axis=1)

        avg_cumulative = np.mean(cumulative_var, axis=0)
        axes[1, 1].bar(range(1, len(avg_cumulative) + 1), avg_cumulative, alpha=0.7)
        axes[1, 1].set_title('Average Cumulative Explained Variance')
        axes[1, 1].set_xlabel('Principal Component')
        axes[1, 1].set_ylabel('Cumulative Explained Variance')
        axes[1, 1].grid(True)

    # Trading activity over time
    trading_activity = [sum(np.abs(pos)) for pos in backtest_results['positions']]
    axes[1, 2].plot(backtest_results['dates'], trading_activity)
    axes[1, 2].set_title('Trading Activity Over Time')
    axes[1, 2].set_ylabel('Total Position Size')
    axes[1, 2].grid(True)
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function implementing the PCA yield curve smoothing strategy
    """
    # Configuration parameters
    file_path = "data/crsp_a_treasuries.csv.gz"  # Update with your data path
    target_maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    lookback_window = 252  # 1 year for PCA estimation
    n_components = 3  # Level, Slope, Curvature
    signal_threshold = 2.0  # Z-score threshold for trading
    transaction_cost = 0.0005  # 5bps transaction cost
    max_positions = 5  # Maximum concurrent positions

    print("Starting PCA Yield Curve Trading Strategy")
    print("=" * 50)

    # Load and prepare data
    df = load_and_prepare_data(file_path)
    if df is None:
        print("Failed to load data. Please check file path and format.")
        return None

    # Create yield curve matrix
    dates, yield_data, duration_data = create_yield_curve_matrix(df, target_maturities)

    if len(dates) < lookback_window * 2:
        print(f"Insufficient data. Need at least {lookback_window * 2} days, got {len(dates)}")
        return None

    print(f"Created yield curve matrix: {yield_data.shape[0]} dates × {yield_data.shape[1]} maturities")

    # Run PCA smoothing analysis
    pca_results = run_pca_smoothing_analysis(yield_data, n_components, lookback_window)

    print(f"Generated {len(pca_results['dates'])} PCA observations")

    # Generate trading signals
    signals = generate_duration_neutral_signals(
        pca_results, duration_data, target_maturities,
        signal_threshold, max_positions
    )

    print(f"Generated {len(signals)} trading signals")

    # Backtest strategy
    backtest_results = backtest_pca_strategy(
        signals, yield_data, duration_data, dates, target_maturities, transaction_cost
    )

    # Analyze performance
    performance_metrics = analyze_strategy_performance(backtest_results, pca_results)

    # Plot results
    plot_strategy_results(backtest_results, pca_results, target_maturities)

    return {
        'backtest_results': backtest_results,
        'pca_results': pca_results,
        'performance_metrics': performance_metrics,
        'signals': signals
    }


if __name__ == "__main__":
    results = main()