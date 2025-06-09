import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


@dataclass
class StrategyConfig:
    pca_window: int = 252  # 1 year rolling window
    signal_threshold: float = 2.0  # Standard deviation threshold for signals
    max_leverage: float = 3.0  # Maximum gross leverage
    vol_target: float = 0.10  # 10% annualized volatility target
    transaction_cost_bps: float = 2.0  # 2 bps transaction costs
    rebalance_frequency: int = 1  # Daily rebalancing
    min_trade_size: float = 0.001  # Minimum position size (1bp duration)


class YieldCurveData:

    def __init__(self, file_path: str = 'combined_curves.csv'):
        self.data = self._load_and_validate_data(file_path)
        self.tenors = self.data.columns.tolist()

    def _load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            df['observation_date'] = pd.to_datetime(df['observation_date'])
            df.set_index('observation_date', inplace=True)

            # Data quality checks
            if df.isnull().sum().sum() > 0:
                print(f"Warning: {df.isnull().sum().sum()} missing values detected")
                df = df.dropna()

            # Check for reasonable yield ranges (0-25%)
            if (df < 0).any().any() or (df > 0.25).any().any():
                print("Warning: Yields outside expected range detected")

            print(f"Loaded yield data: {df.shape[0]} observations, {df.shape[1]} tenors")
            print(f"Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            raise ValueError(f"Error loading yield data: {e}")


class PCAFactorModel:

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()

    def fit(self, yield_data: np.ndarray) -> None:
        scaled_data = self.scaler.fit_transform(yield_data)
        self.pca.fit(scaled_data)

    def compute_dislocations(
            self,
            current_yields: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        scaled_current = self.scaler.transform(current_yields.reshape(1, -1))
        pca_projection = self.pca.transform(scaled_current)
        reconstructed_scaled = self.pca.inverse_transform(pca_projection)
        reconstructed_yields = self.scaler.inverse_transform(reconstructed_scaled).flatten()

        deviations = current_yields - reconstructed_yields
        std_dev = np.std(deviations)
        dislocations = deviations / std_dev if std_dev > 0 else np.zeros_like(deviations)

        explained_variance = self.pca.explained_variance_ratio_.sum()

        return reconstructed_yields, dislocations, explained_variance

    def get_factor_loadings(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i + 1}' for i in range(self.n_components)]
        )


class PositionManager:

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.position_history = []

    def compute_transaction_costs(
            self,
            current_positions: np.ndarray,
            target_positions: np.ndarray
    ) -> float:
        position_changes = np.abs(target_positions - current_positions)
        total_turnover = np.sum(position_changes)
        return total_turnover * self.config.transaction_cost_bps / 10000

    def compute_positions(self,
                          dislocations: np.ndarray,
                          current_vol: float,
                          current_positions: np.ndarray) -> np.ndarray:
        """
        Compute target positions based on dislocations and risk controls

        Strategy:
        - Long undervalued (negative dislocation) tenors
        - Short overvalued (positive dislocation) tenors
        - Scale by volatility targeting
        - Apply leverage constraints
        """
        # Generate raw signals
        raw_signals = np.where(
            np.abs(dislocations) > self.config.signal_threshold,
            -dislocations,  # Negative dislocation -> positive signal (long)
            0.0
        )

        # Volatility scaling
        vol_scalar = self.config.vol_target / max(current_vol, 0.01)
        scaled_signals = raw_signals * vol_scalar

        # Apply leverage constraint
        gross_exposure = np.sum(np.abs(scaled_signals))
        if gross_exposure > self.config.max_leverage:
            scaled_signals *= self.config.max_leverage / gross_exposure

        # Apply minimum trade size filter
        scaled_signals = np.where(
            np.abs(scaled_signals) < self.config.min_trade_size,
            0.0,
            scaled_signals
        )

        return scaled_signals


class YieldCurveArbitrageStrategy:

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.factor_model = PCAFactorModel()
        self.position_manager = PositionManager(self.config)

    def backtest(self, yield_data: YieldCurveData) -> Dict:
        df = yield_data.data
        tenors = yield_data.tenors
        start_idx = self.config.pca_window

        results = []
        positions = np.zeros(len(tenors))
        portfolio_value = 1.0

        print(f"Starting backtest with {len(df) - start_idx} periods...")

        for i in range(start_idx, len(df)):
            current_date = df.index[i]

            window_data = df.iloc[i - self.config.pca_window:i].values
            self.factor_model.fit(window_data)

            current_yields = df.iloc[i].values
            reconstructed, dislocations, explained_var = self.factor_model.compute_dislocations(current_yields)

            if i > start_idx + 21:
                recent_returns = [r['daily_pnl'] for r in results[-21:]]
                current_vol = np.std(recent_returns) * np.sqrt(252)
            else:
                current_vol = self.config.vol_target

            target_positions = self.position_manager.compute_positions(
                dislocations, current_vol, positions
            )

            transaction_costs = self.position_manager.compute_transaction_costs(
                positions, target_positions
            )

            positions = target_positions.copy()

            if i < len(df) - 1:
                yield_changes = df.iloc[i + 1].values - current_yields
                daily_pnl = -np.sum(positions * yield_changes * 10000)  # Convert to bps
                daily_pnl -= transaction_costs * 10000  # Subtract transaction costs
                portfolio_value *= (1 + daily_pnl / 10000)
            else:
                daily_pnl = 0

            result = {
                'date': current_date,
                'portfolio_value': portfolio_value,
                'daily_pnl': daily_pnl,
                'gross_exposure': np.sum(np.abs(positions)),
                'transaction_costs': transaction_costs,
                'explained_variance': explained_var,
                'max_dislocation': np.max(np.abs(dislocations)),
                'active_signals': np.sum(np.abs(positions) > 0)
            }

            for j, tenor in enumerate(tenors):
                result.update({
                    f'{tenor}_dislocation': dislocations[j],
                    f'{tenor}_position': positions[j],
                    f'{tenor}_yield': current_yields[j]
                })

            results.append(result)

            if i % 252 == 0:
                print(f"Processed {i - start_idx} periods, Portfolio value: {portfolio_value:.3f}")

        return self._compile_results(results, tenors)

    def _compile_results(self, results: List[Dict]) -> Dict:
        df = pd.DataFrame(results).set_index('date')

        df['daily_return'] = df['daily_pnl'] / 10000
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

        total_return = df['cumulative_return'].iloc[-1]
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1
        volatility = df['daily_return'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(df['portfolio_value'])

        avg_gross_exposure = df['gross_exposure'].mean()
        avg_transaction_costs = df['transaction_costs'].mean()
        avg_active_signals = df['active_signals'].mean()

        return {
            'performance_df': df,
            'metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_gross_exposure': avg_gross_exposure,
                'avg_transaction_costs': avg_transaction_costs,
                'avg_active_signals': avg_active_signals
            },
            'config': self.config
        }

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()


class PerformanceAnalyzer:

    @staticmethod
    def generate_report(backtest_results: Dict) -> None:
        df = backtest_results['performance_df']
        metrics = backtest_results['metrics']

        print(f"\nStrategy Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Total Observations: {len(df)}")

        print(f"\nPerformance Metrics:")
        print(f"  Total Return:       {metrics['total_return']:.2%}")
        print(f"  Annualized Return:  {metrics['annualized_return']:.2%}")
        print(f"  Volatility:         {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:       {metrics['max_drawdown']:.2%}")

        print(f"\nStrategy Statistics:")
        print(f"  Avg Gross Exposure: {metrics['avg_gross_exposure']:.2f}")
        print(f"  Avg Transaction Costs: {metrics['avg_transaction_costs']:.4f}")
        print(f"  Avg Active Signals: {metrics['avg_active_signals']:.1f}")

    @staticmethod
    def plot_performance(backtest_results: Dict) -> None:
        df = backtest_results['performance_df']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Portfolio value
        ax1.plot(df.index, df['portfolio_value'])
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True)

        # Daily returns
        ax2.hist(df['daily_return'], bins=50, alpha=0.7)
        ax2.set_title('Distribution of Daily Returns')
        ax2.set_xlabel('Daily Return')
        ax2.grid(True)

        # Gross exposure
        ax3.plot(df.index, df['gross_exposure'])
        ax3.set_title('Gross Exposure Over Time')
        ax3.set_ylabel('Gross Exposure')
        ax3.grid(True)

        # Max dislocation
        ax4.plot(df.index, df['max_dislocation'])
        ax4.axhline(y=2.0, color='r', linestyle='--', label='Signal Threshold')
        ax4.set_title('Maximum Yield Curve Dislocation')
        ax4.set_ylabel('Max |Dislocation| (Std Dev)')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()


def main():

    config = StrategyConfig()

    yield_data = YieldCurveData('combined_curves.csv')

    strategy = YieldCurveArbitrageStrategy(config)
    results = strategy.backtest(yield_data)

    PerformanceAnalyzer.generate_report(results)
    PerformanceAnalyzer.plot_performance(results)

    print(f"\nStrategy Configuration Used:")
    print(f"  PCA Window: {config.pca_window} days")
    print(f"  Signal Threshold: {config.signal_threshold} std dev")
    print(f"  Max Leverage: {config.max_leverage}")
    print(f"  Vol Target: {config.vol_target:.1%}")
    print(f"  Transaction Costs: {config.transaction_cost_bps} bps")


if __name__ == '__main__':
    main()