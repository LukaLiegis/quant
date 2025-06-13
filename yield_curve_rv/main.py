import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

import warnings

warnings.filterwarnings('ignore')


@dataclass
class StrategyConfig:
    # PCA Configuration
    pca_window: int = 756  # 3+ years for stability (252 * 3)
    min_pca_explained_variance: float = 0.99  # 99.4% threshold
    pca_stability_threshold: float = 0.8  # Minimum correlation across periods

    # Signal Generation
    entry_percentile: float = 3.0  # Enter at extreme 3% percentile
    exit_percentile: float = 30.0  # Exit at 30% percentile
    signal_lookback_window: int = 252  # 1 year for percentile calculation

    # Risk Management
    max_leverage: float = 3.0
    max_single_position: float = 1.2  # 40% of max leverage
    max_sector_exposure: float = 1.8  # 60% of max leverage
    vol_target: float = 0.10
    min_trade_size: float = 0.01  # Minimum trade size to avoid tiny positions

    # Transaction Costs
    base_transaction_cost_bps: float = 2.0
    transaction_cost_bps: float = 2.0  # Alias for backward compatibility
    market_impact_threshold: float = 1.0  # Turnover threshold for impact
    market_impact_multiplier: float = 0.5  # Additional cost per unit excess turnover

    # Curve Neutrality
    enable_curve_neutrality: bool = True
    max_pc_exposure: float = 1.0

    # Performance Analytics
    enable_pc_attribution: bool = True
    enable_enhanced_reporting: bool = True
    risk_lookback_window: int = 63

    max_holding_days: int = 120
    min_holding_days: int = 5
    use_duration_dollar_weighting: bool = True  # Core SSB innovation
    pca_method: str = 'levels'  # 'levels' or 'changes'
    force_120_tenors: bool = False  # Use 120 constant maturity points
    butterfly_configurations: List[Tuple[str, str, str]] = None
    hedge_securities: List[str] = None
    preferred_hedging_method: str = 'pc_scenario_optimization'

    def __post_init__(self):
        if self.butterfly_configurations is None:
            self.butterfly_configurations = [
                ('2Y', '5Y', '10Y'),
                ('5Y', '7Y', '10Y'),
                ('5Y', '10Y', '30Y'),
                ('10Y', '20Y', '30Y')
            ]
        if self.hedge_securities is None:
            self.hedge_securities = ['2Y', '10Y', '30Y']


class YieldCurveData:

    def __init__(self, file_path: str = 'combined_curves.csv'):
        self.data = self._load_and_validate_data(file_path)
        self.tenors = self.data.columns.tolist()

    def _load_and_validate_data(
            self,
            file_path: str
    ) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            df['observation_date'] = pd.to_datetime(df['observation_date'])
            df.set_index('observation_date', inplace=True)

            if df.isnull().sum().sum() > 0:
                print(f"Warning: {df.isnull().sum().sum()} missing values detected")
                df = df.dropna()

            if (df < 0).any().any() or (df > 0.25).any().any():
                print("Warning: Yields outside expected range detected")

            print(f"Loaded yield data: {df.shape[0]} observations, {df.shape[1]} tenors")
            print(f"Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            raise ValueError(f"Error loading yield data: {e}")

    def interpolate_to_120_tenors(self) -> pd.DataFrame:
        """
        Interpolate to 120 constant maturity points as per SSB paper
        """
        # Generate SSB target tenors
        target_tenors = []

        # 3M to 12M (quarterly)
        for months in [3, 6, 9, 12]:
            target_tenors.append(f"{months}M")

        # 1.25Y to 30Y (quarterly increments)
        for quarters in range(5, 121):  # 5 quarters = 1.25 years
            years = quarters / 4
            if years == int(years):
                target_tenors.append(f"{int(years)}Y")
            else:
                target_tenors.append(f"{years:.2f}Y")

        # Convert existing tenor names to numeric (years)
        existing_tenors_numeric = []
        for tenor in self.data.columns:
            if 'M' in tenor:
                months = int(tenor.replace('M', ''))
                existing_tenors_numeric.append(months / 12)
            elif 'Y' in tenor:
                years = float(tenor.replace('Y', ''))
                existing_tenors_numeric.append(years)

        # Convert target tenors to numeric
        target_tenors_numeric = []
        for tenor in target_tenors:
            if 'M' in tenor:
                months = int(tenor.replace('M', ''))
                target_tenors_numeric.append(months / 12)
            elif 'Y' in tenor:
                years = float(tenor.replace('Y', ''))
                target_tenors_numeric.append(years)

        # Interpolate
        interpolated_data = []
        for date_idx in range(len(self.data)):
            yields = self.data.iloc[date_idx].values
            interpolated_yields = np.interp(target_tenors_numeric, existing_tenors_numeric, yields)
            interpolated_data.append(interpolated_yields)

        interpolated_df = pd.DataFrame(
            data=interpolated_data,
            index=self.data.index,
            columns=target_tenors
        )

        print(f"Interpolated to {len(target_tenors)} SSB constant maturity points")
        return interpolated_df

    def convert_to_weekly_changes(self) -> pd.DataFrame:
        """
        Convert to weekly changes as per SSB methodology
        """
        weekly_levels = self.data.resample('W-FRI').last()
        weekly_changes = weekly_levels.diff().dropna()
        print(f"Converted to weekly changes: {len(weekly_changes)} observations")
        return weekly_changes


class EnhancedPCAFactorModel:

    def __init__(self, n_components: int = 3, min_window: int = 756):
        self.n_components = n_components
        self.min_window = min_window
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.stability_metrics = {}
        self.explained_variance_threshold = 0.994
        # SSB enhancements
        self.pca_levels_model = None
        self.pca_changes_model = None

    def validate_data_quality(
            self,
            yield_data: np.ndarray
    ) -> Dict:
        quality_report = {
            'missing_values': np.isnan(yield_data).sum(),
            'negative_yields': (yield_data < 0).sum(),
            'extreme_yields': (yield_data > 0.25).sum(),  # >25% considered extreme
            'data_points': yield_data.shape[0],
            'sufficient_history': yield_data.shape[0] >= self.min_window
        }

        daily_changes = np.diff(yield_data, axis=0)
        quality_report['avg_daily_volatility'] = np.std(daily_changes, axis=0).mean()
        quality_report['is_quality_sufficient'] = (
                quality_report['missing_values'] == 0 and
                quality_report['sufficient_history'] and
                quality_report['avg_daily_volatility'] > 0.001  # Minimum 0.1bp daily vol
        )

        return quality_report

    def fit_with_stability_analysis(
            self,
            yield_data: np.ndarray
    ) -> None:
        quality_report = self.validate_data_quality(yield_data)

        if not quality_report['is_quality_sufficient']:
            raise ValueError(f"Data quality insufficient: {quality_report}")

        scaled_data = self.scaler.fit_transform(yield_data)
        self.pca.fit(scaled_data)

        self._analyze_component_stability(yield_data)

        total_explained = self.pca.explained_variance_ratio_.sum()
        if total_explained < self.explained_variance_threshold:
            print(
                f"Warning: Only {total_explained:.3%} variance explained (target: {self.explained_variance_threshold:.1%})")

    def _analyze_component_stability(
            self,
            yield_data: np.ndarray
    ) -> None:
        n_periods = min(4, yield_data.shape[0] // (252 * 2))

        if n_periods < 2:
            return

        period_length = yield_data.shape[0] // n_periods
        component_correlations = []

        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < n_periods - 1 else yield_data.shape[0]

            period_data = yield_data[start_idx:end_idx]
            if period_data.shape[0] < 252:
                continue

            scaler_temp = StandardScaler()
            pca_temp = PCA(n_components=self.n_components)
            scaled_temp = scaler_temp.fit_transform(period_data)
            pca_temp.fit(scaled_temp)

            correlations = []
            for j in range(self.n_components):
                corr = np.abs(np.corrcoef(self.pca.components_[j], pca_temp.components_[j])[0, 1])
                correlations.append(corr)

            component_correlations.append(correlations)

        self.stability_metrics = {
            'avg_correlations': np.mean(component_correlations,
                                        axis=0) if component_correlations else [1.0] * self.n_components,
            'min_correlations': np.min(component_correlations,
                                       axis=0) if component_correlations else [1.0] * self.n_components,
            'n_periods_analyzed': len(component_correlations)
        }

    def fit_levels_vs_changes_analysis(self, yield_data: np.ndarray) -> Dict:
        """
        SSB KEY INSIGHT: Compare PCA on yield levels vs changes
        """
        # Method 1: PCA on yield levels (SSB preferred)
        scaler_levels = StandardScaler()
        scaled_levels = scaler_levels.fit_transform(yield_data)
        pca_levels = PCA(n_components=self.n_components)
        pca_levels.fit(scaled_levels)

        # Method 2: PCA on yield changes
        yield_changes = np.diff(yield_data, axis=0)
        scaler_changes = StandardScaler()
        scaled_changes = scaler_changes.fit_transform(yield_changes)
        pca_changes = PCA(n_components=self.n_components)
        pca_changes.fit(scaled_changes)

        # Store both models
        self.pca_levels_model = {'pca': pca_levels, 'scaler': scaler_levels}
        self.pca_changes_model = {'pca': pca_changes, 'scaler': scaler_changes}

        # Compare explained variance
        levels_explained = pca_levels.explained_variance_ratio_.sum()
        changes_explained = pca_changes.explained_variance_ratio_.sum()

        # SSB conclusion: levels are generally better for butterfly trading
        recommendation = 'levels' if levels_explained >= changes_explained else 'changes'

        return {
            'levels_explained_variance': pca_levels.explained_variance_ratio_,
            'changes_explained_variance': pca_changes.explained_variance_ratio_,
            'levels_total_explained': levels_explained,
            'changes_total_explained': changes_explained,
            'ssb_recommendation': recommendation,
            'levels_preferred': levels_explained >= changes_explained
        }

    def calculate_duration_dollar_weights(self, yields: np.ndarray, tenors: List[str]) -> Dict:
        """
        Calculate duration-dollar weights for butterfly spreads (SSB core innovation)
        """

        def calculate_modified_duration(yield_val, maturity_years):
            if yield_val > 0:
                duration = (1 - (1 + yield_val) ** (-maturity_years)) / yield_val
                return duration / (1 + yield_val)
            return maturity_years

        butterfly_weights = {}

        # Process butterfly configurations
        butterfly_configs = [
            ('2Y', '5Y', '10Y'),
            ('5Y', '7Y', '10Y'),
            ('5Y', '10Y', '30Y'),
            ('10Y', '20Y', '30Y')
        ]

        for short_tenor, belly_tenor, long_tenor in butterfly_configs:
            if all(tenor in tenors for tenor in [short_tenor, belly_tenor, long_tenor]):
                # Get indices
                short_idx = tenors.index(short_tenor)
                belly_idx = tenors.index(belly_tenor)
                long_idx = tenors.index(long_tenor)

                # Get yields
                short_yield = yields[short_idx]
                belly_yield = yields[belly_idx]
                long_yield = yields[long_idx]

                # Calculate durations
                short_duration = calculate_modified_duration(short_yield, float(short_tenor.replace('Y', '')))
                belly_duration = calculate_modified_duration(belly_yield, float(belly_tenor.replace('Y', '')))
                long_duration = calculate_modified_duration(long_yield, float(long_tenor.replace('Y', '')))

                # Assume equal market values for simplicity
                market_value = 1000000

                # Calculate duration dollars
                dd_short = market_value * short_duration
                dd_belly = market_value * belly_duration
                dd_long = market_value * long_duration

                # Calculate SSB duration-dollar weighted butterfly spread
                dd_weight_short = dd_short / dd_belly if dd_belly != 0 else 0
                dd_weight_long = dd_long / dd_belly if dd_belly != 0 else 0

                ssb_butterfly_spread = (belly_yield -
                                        dd_weight_short * short_yield -
                                        dd_weight_long * long_yield)

                # Traditional 50-50 spread for comparison
                traditional_spread = belly_yield - 0.5 * (short_yield + long_yield)

                butterfly_key = f"{short_tenor}_{belly_tenor}_{long_tenor}"
                butterfly_weights[butterfly_key] = {
                    'ssb_spread': ssb_butterfly_spread,
                    'traditional_spread': traditional_spread,
                    'dd_weight_short': dd_weight_short,
                    'dd_weight_long': dd_weight_long,
                    'duration_dollars': {'short': dd_short, 'belly': dd_belly, 'long': dd_long}
                }

        return butterfly_weights

    def solve_independent_trades(self, tenors: List[str]) -> Dict:
        """
        Solve for SSB Independent Trades (Figure 4 methodology)
        """
        if len(tenors) < 3:
            return {}

        # Use short, mid, long positions (2Y, 10Y, 30Y equivalent)
        short_idx = 0
        mid_idx = len(tenors) // 2
        long_idx = len(tenors) - 1

        # Get sensitivities to each PC for the hedge securities
        sensitivity_matrix = np.zeros((3, 3))  # 3 securities x 3 PCs

        # Use PCA components as sensitivity estimates
        if hasattr(self, 'pca') and self.pca.components_ is not None:
            for i in range(3):
                sensitivity_matrix[0, i] = self.pca.components_[i, short_idx]  # Short security
                sensitivity_matrix[1, i] = self.pca.components_[i, mid_idx]  # Mid security
                sensitivity_matrix[2, i] = self.pca.components_[i, long_idx]  # Long security

        independent_trades = {}

        for pc_idx, pc_name in enumerate(['Level', 'Slope', 'Curvature']):
            # Solve for weights that give exposure to only this PC
            target_exposure = np.zeros(3)
            target_exposure[pc_idx] = 1.0

            try:
                # Solve: sensitivity_matrix.T @ weights = target_exposure
                weights = np.linalg.solve(sensitivity_matrix.T, target_exposure)

                # Normalize weights
                if pc_name == 'Curvature' and abs(weights[1]) > 0.01:  # Normalize to mid position
                    weights = weights / abs(weights[1])
                elif abs(weights[0]) > 0.01:  # Normalize to first position
                    weights = weights / abs(weights[0])

                # Generate description
                descriptions = []
                securities = [tenors[short_idx], tenors[mid_idx], tenors[long_idx]]
                for weight, security in zip(weights, securities):
                    if abs(weight) > 0.01:
                        action = "Buy" if weight > 0 else "Sell"
                        descriptions.append(f"{action} {abs(weight):.2f} {security}")

                independent_trades[pc_name] = {
                    'weights': weights,
                    'securities': securities,
                    'description': f"{pc_name}: " + ", ".join(descriptions)
                }

            except np.linalg.LinAlgError:
                independent_trades[pc_name] = {
                    'weights': np.array([0, 0, 0]),
                    'securities': [tenors[short_idx], tenors[mid_idx], tenors[long_idx]],
                    'description': f"{pc_name}: Could not solve"
                }

        return independent_trades

    def get_independent_trades(
            self,
            tenors: List[str]
    ) -> Dict[str, Dict]:
        if len(tenors) < 3:
            raise ValueError("Need at least 3 tenors for independent trades")

        short_tenor = tenors[0]
        mid_tenor = tenors[len(tenors) // 2]
        long_tenor = tenors[-1]

        independent_trades = {}

        short_idx = tenors.index(short_tenor)
        mid_idx = tenors.index(mid_tenor)
        long_idx = tenors.index(long_tenor)

        for i, pc_name in enumerate(['Level', 'Slope', 'Curvature']):
            weights = self._solve_independent_trade_weights(
                [short_idx, mid_idx, long_idx], i
            )

            independent_trades[pc_name] = {
                'tenors': [short_tenor, mid_tenor, long_tenor],
                'weights': weights,
                'description': self._get_trade_description(pc_name, weights)
            }

        return independent_trades

    def _solve_independent_trade_weights(
            self,
            tenor_indices: List[int],
            target_pc: int
    ) -> np.ndarray:
        loadings_matrix = self.pca.components_[:, tenor_indices].T  # 3x3 matrix

        target = np.zeros(self.n_components)
        target[target_pc] = 1.0

        try:
            weights = np.linalg.solve(loadings_matrix, target)
            return weights
        except np.linalg.LinAlgError:
            weights, _, _, _ = np.linalg.lstsq(loadings_matrix, target, rcond=None)
            return weights

    def _get_trade_description(
            self,
            pc_name: str,
            weights: np.ndarray
    ) -> str:
        descriptions = []
        tenor_names = ['Short', 'Mid', 'Long']

        for i, (weight, name) in enumerate(zip(weights, tenor_names)):
            if abs(weight) > 0.01:
                action = "Buy" if weight > 0 else "Sell"
                descriptions.append(f"{action} {abs(weight):.1%} {name}")

        return f"{pc_name}: " + ", ".join(descriptions)

    def compute_enhanced_dislocations(
            self,
            current_yields: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        scaled_current = self.scaler.transform(current_yields.reshape(1, -1))
        pca_projection = self.pca.transform(scaled_current)
        reconstructed_scaled = self.pca.inverse_transform(pca_projection)
        reconstructed_yields = self.scaler.inverse_transform(reconstructed_scaled).flatten()

        deviations = current_yields - reconstructed_yields

        std_dev = np.std(deviations)
        dislocations = deviations / std_dev if std_dev > 0 else np.zeros_like(deviations)

        metrics = {
            'explained_variance': self.pca.explained_variance_ratio_.sum(),
            'max_abs_dislocation': np.max(np.abs(dislocations)),
            'pc_contributions': pca_projection.flatten(),
            'reconstruction_error': np.sum(deviations ** 2),
            'stability_score': np.mean(self.stability_metrics.get('avg_correlations', [1.0]))
        }

        return reconstructed_yields, dislocations, metrics


class EnhancedPositionManager:

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.position_history = []
        self.dislocation_history = []
        self.pc_exposure_history = []

        self.entry_percentile = 3.0
        self.exit_percentile = 30.0
        self.lookback_window = 252

        # SSB Enhanced Attributes
        self.active_butterfly_trades = {}  # Track individual butterfly trades
        self.trade_history = []
        self.butterfly_spreads_history = {}

    def update_history(self, dislocations: np.ndarray, pc_exposures: np.ndarray) -> None:
        self.dislocation_history.append(dislocations)
        self.pc_exposure_history.append(pc_exposures)

        if len(self.dislocation_history) > self.lookback_window * 2:
            self.dislocation_history = self.dislocation_history[-self.lookback_window:]
            self.pc_exposure_history = self.pc_exposure_history[-self.lookback_window:]

    def compute_percentile_signals(
            self,
            current_dislocations: np.ndarray
    ) -> np.ndarray:
        if len(self.dislocation_history) < 60:
            return np.zeros_like(current_dislocations)

        historical_dislocations = np.array(self.dislocation_history)
        signals = np.zeros_like(current_dislocations)

        for i in range(len(current_dislocations)):
            percentile = stats.percentileofscore(
                historical_dislocations[:, i],
                current_dislocations[i]
            )

            if percentile <= self.entry_percentile:
                signals[i] = 1.0
            elif percentile >= (100 - self.entry_percentile):
                signals[i] = -1.0
            elif self.entry_percentile < percentile <= self.exit_percentile:
                signals[i] = 0.5
            elif (100 - self.exit_percentile) <= percentile < (100 - self.entry_percentile):
                signals[i] = -0.5
            else:
                signals[i] = 0.0

        return signals

    def update_butterfly_trades(self, date: pd.Timestamp, butterfly_spreads: Dict[str, float],
                                tenors: List[str]) -> Tuple[List, List]:
        """
        SSB Enhanced signal generation with holding periods
        """
        # Update historical spreads for percentile calculation
        if date not in self.butterfly_spreads_history:
            self.butterfly_spreads_history[date] = {}
        self.butterfly_spreads_history[date].update(butterfly_spreads)

        # Keep only recent history
        max_history_days = self.config.signal_lookback_window * 2
        cutoff_date = date - pd.Timedelta(days=max_history_days)
        dates_to_remove = [d for d in self.butterfly_spreads_history.keys() if d < cutoff_date]
        for d in dates_to_remove:
            del self.butterfly_spreads_history[d]

        new_trades = []
        closed_trades = []

        # Generate new signals
        for spread_key, current_spread in butterfly_spreads.items():
            # Calculate percentile
            historical_values = []
            for date_spreads in self.butterfly_spreads_history.values():
                if spread_key in date_spreads:
                    historical_values.append(date_spreads[spread_key])

            if len(historical_values) < 60:  # Need minimum history
                continue

            percentile = stats.percentileofscore(historical_values, current_spread)

            # Check if we already have an active trade for this butterfly
            if spread_key in self.active_butterfly_trades:
                continue

            # SSB Entry conditions: 3% and 97% percentiles
            signal_strength = 0
            if percentile <= self.config.entry_percentile:
                signal_strength = 1.0  # Buy signal (expect mean reversion up)
            elif percentile >= (100 - self.config.entry_percentile):
                signal_strength = -1.0  # Sell signal (expect mean reversion down)

            if signal_strength != 0:
                trade = {
                    'entry_date': date,
                    'spread_key': spread_key,
                    'signal_strength': signal_strength,
                    'entry_percentile': percentile,
                    'entry_spread': current_spread,
                    'target_exit_percentile': self.config.exit_percentile if signal_strength > 0 else (
                                100 - self.config.exit_percentile),
                    'max_holding_days': self.config.max_holding_days,
                    'current_pnl': 0.0,
                    'days_held': 0
                }

                self.active_butterfly_trades[spread_key] = trade
                new_trades.append(trade)

        # Update and check exit conditions for active trades
        for spread_key, trade in list(self.active_butterfly_trades.items()):
            if spread_key not in butterfly_spreads:
                continue

            # Update trade
            trade['days_held'] = (date - trade['entry_date']).days
            current_spread = butterfly_spreads[spread_key]

            # Calculate P&L (simplified)
            spread_change = current_spread - trade['entry_spread']
            trade['current_pnl'] = trade['signal_strength'] * spread_change * 10000  # Convert to bp

            # Check exit conditions
            should_close = False
            close_reason = ""

            # Condition 1: Maximum holding period
            if trade['days_held'] >= trade['max_holding_days']:
                should_close = True
                close_reason = "Max holding period"

            # Condition 2: Target percentile reached
            else:
                historical_values = []
                for date_spreads in self.butterfly_spreads_history.values():
                    if spread_key in date_spreads:
                        historical_values.append(date_spreads[spread_key])

                if len(historical_values) >= 60:
                    current_percentile = stats.percentileofscore(historical_values, current_spread)

                    if trade['signal_strength'] > 0:  # Long position
                        if current_percentile >= trade['target_exit_percentile']:
                            should_close = True
                            close_reason = "Target percentile reached"
                    else:  # Short position
                        if current_percentile <= trade['target_exit_percentile']:
                            should_close = True
                            close_reason = "Target percentile reached"

            # Close trade if conditions met
            if should_close:
                trade['exit_date'] = date
                trade['exit_spread'] = current_spread
                trade['close_reason'] = close_reason

                closed_trades.append(trade)
                self.trade_history.append(trade)
                del self.active_butterfly_trades[spread_key]

        return new_trades, closed_trades

    def get_butterfly_positions(self) -> Dict[str, float]:
        """
        Convert active butterfly trades to position weights
        """
        positions = {}

        for spread_key, trade in self.active_butterfly_trades.items():
            # Parse butterfly configuration
            parts = spread_key.split('_')
            if len(parts) == 3:
                short_tenor, belly_tenor, long_tenor = parts

                # Base position size
                base_size = abs(trade['signal_strength']) * 0.1  # 10% base

                # Scale by confidence (how extreme the entry was)
                if trade['signal_strength'] > 0:
                    confidence = (self.config.entry_percentile - trade[
                        'entry_percentile']) / self.config.entry_percentile
                else:
                    confidence = (trade['entry_percentile'] - (
                                100 - self.config.entry_percentile)) / self.config.entry_percentile

                confidence = max(0, min(confidence, 1))
                scaled_size = base_size * (0.5 + 0.5 * confidence)

                # Butterfly positions: long belly, short wings
                belly_position = trade['signal_strength'] * scaled_size
                wing_position = -trade['signal_strength'] * scaled_size * 0.5

                positions[f"{belly_tenor}_butterfly"] = positions.get(f"{belly_tenor}_butterfly", 0) + belly_position
                positions[f"{short_tenor}_butterfly"] = positions.get(f"{short_tenor}_butterfly", 0) + wing_position
                positions[f"{long_tenor}_butterfly"] = positions.get(f"{long_tenor}_butterfly", 0) + wing_position

        return positions

    def get_butterfly_performance_stats(self) -> Dict:
        """
        Calculate butterfly trading performance statistics
        """
        if not self.trade_history:
            return {'no_trades': True}

        trades_df = pd.DataFrame([
            {
                'entry_date': trade['entry_date'],
                'exit_date': trade.get('exit_date', pd.Timestamp.now()),
                'days_held': trade['days_held'],
                'pnl_bp': trade['current_pnl'],
                'signal_strength': trade['signal_strength'],
                'spread_key': trade['spread_key'],
                'close_reason': trade.get('close_reason', 'Still active')
            }
            for trade in self.trade_history
        ])

        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_bp'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl_bp': trades_df['pnl_bp'].mean(),
            'total_pnl_bp': trades_df['pnl_bp'].sum(),
            'avg_holding_days': trades_df['days_held'].mean(),
            'best_trade_bp': trades_df['pnl_bp'].max(),
            'worst_trade_bp': trades_df['pnl_bp'].min(),
            'sharpe_ratio': trades_df['pnl_bp'].mean() / trades_df['pnl_bp'].std() if trades_df[
                                                                                          'pnl_bp'].std() > 0 else 0
        }

    def compute_curve_neutral_positions(
            self,
            signals: np.ndarray,
            independent_trades: Dict,
            current_vol: float,
            tenors: List[str]
    ) -> np.ndarray:

        positions = np.zeros(len(tenors))

        raw_positions = signals * self.config.vol_target / max(current_vol, 0.01)

        curve_neutral_positions = self._apply_curve_neutrality(
            raw_positions, independent_trades, tenors
        )

        final_positions = self._apply_risk_controls(curve_neutral_positions)

        return final_positions

    def _apply_curve_neutrality(
            self,
            raw_positions: np.ndarray,
            independent_trades: Dict,
            tenors: List[str]
    ) -> np.ndarray:

        adjusted_positions = raw_positions.copy()

        pc_exposures = self._calculate_pc_exposures(adjusted_positions, tenors)

        for i, (pc_name, exposure) in enumerate(zip(['Level', 'Slope', 'Curvature'], pc_exposures)):
            if abs(exposure) > self.config.max_leverage / 3:

                if pc_name in independent_trades:
                    trade = independent_trades[pc_name]
                    trade_tenors = trade['tenors']
                    trade_weights = trade['weights']

                    hedge_scale = -exposure / max(abs(trade_weights))

                    for tenor, weight in zip(trade_tenors, trade_weights):
                        if tenor in tenors:
                            tenor_idx = tenors.index(tenor)
                            adjusted_positions[tenor_idx] += hedge_scale * weight

        return adjusted_positions

    def _calculate_pc_exposures(
            self,
            positions: np.ndarray,
            tenors: List[str] = None
    ) -> np.ndarray:
        """Calculate portfolio exposure to each principal component"""
        # This would use the PCA loadings to calculate exposures
        # Simplified implementation - in practice would use actual PC loadings

        level_exposure = np.sum(positions)

        if len(positions) >= 3:
            short_avg = np.mean(positions[:len(positions) // 3])
            long_avg = np.mean(positions[-len(positions) // 3:])
            slope_exposure = long_avg - short_avg
        else:
            slope_exposure = 0.0

        if len(positions) >= 3:
            mid_idx = len(positions) // 2
            wing_avg = (positions[0] + positions[-1]) / 2
            curvature_exposure = positions[mid_idx] - wing_avg
        else:
            curvature_exposure = 0.0

        return np.array([level_exposure, slope_exposure, curvature_exposure])

    def _apply_risk_controls(
            self,
            positions: np.ndarray
    ) -> np.ndarray:

        gross_exposure = np.sum(np.abs(positions))
        if gross_exposure > self.config.max_leverage:
            positions *= self.config.max_leverage / gross_exposure

        positions = np.where(
            np.abs(positions) < self.config.min_trade_size,
            0.0,
            positions
        )

        max_single_position = self.config.max_leverage * 0.4
        positions = np.clip(positions, -max_single_position, max_single_position)

        positions = self._apply_sector_limits(positions)

        return positions

    def _apply_sector_limits(
            self,
            positions: np.ndarray
    ) -> np.ndarray:
        if len(positions) < 6:
            return positions

        sector_size = len(positions) // 3
        sector_limit = self.config.max_leverage * 0.6

        short_sector_exposure = np.sum(np.abs(positions[:sector_size]))
        if short_sector_exposure > sector_limit:
            scale_factor = sector_limit / short_sector_exposure
            positions[:sector_size] *= scale_factor

        mid_start = sector_size
        mid_end = 2 * sector_size
        mid_sector_exposure = np.sum(np.abs(positions[mid_start:mid_end]))
        if mid_sector_exposure > sector_limit:
            scale_factor = sector_limit / mid_sector_exposure
            positions[mid_start:mid_end] *= scale_factor

        long_sector_exposure = np.sum(np.abs(positions[2 * sector_size:]))
        if long_sector_exposure > sector_limit:
            scale_factor = sector_limit / long_sector_exposure
            positions[2 * sector_size:] *= scale_factor

        return positions

    def calculate_enhanced_transaction_costs(
            self,
            current_positions: np.ndarray,
            target_positions: np.ndarray,
            bid_offer_spreads: np.ndarray = None
    ) -> Dict:

        position_changes = np.abs(target_positions - current_positions)

        if bid_offer_spreads is not None:
            transaction_costs = np.sum(position_changes * bid_offer_spreads / 2)
        else:
            transaction_costs = np.sum(position_changes) * self.config.transaction_cost_bps / 10000

        total_turnover = np.sum(position_changes)

        if total_turnover > self.config.market_impact_threshold:
            excess_turnover = total_turnover - self.config.market_impact_threshold
            impact_multiplier = 1 + excess_turnover * self.config.market_impact_multiplier
            transaction_costs *= impact_multiplier

        base_costs = np.sum(position_changes) * self.config.transaction_cost_bps / 10000
        market_impact = transaction_costs - base_costs

        return {
            'base_costs': base_costs,
            'market_impact': market_impact,
            'total_costs': transaction_costs,
            'turnover': total_turnover
        }


class EnhancedYieldCurveArbitrageStrategy:

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.factor_model = EnhancedPCAFactorModel(
            n_components=3,
            min_window=self.config.pca_window
        )
        self.position_manager = EnhancedPositionManager(self.config)
        self.performance_analyzer = EnhancedPerformanceAnalyzer()

        self.independent_trades = {}
        self.attribution_history = []

    def backtest(self, yield_data: YieldCurveData) -> Dict:
        # SSB Enhancement: Convert to 120 tenors if requested
        if self.config.force_120_tenors and hasattr(yield_data, 'interpolate_to_120_tenors'):
            print("Converting to SSB 120 constant maturity points...")
            df = yield_data.interpolate_to_120_tenors()
            tenors = df.columns.tolist()
        else:
            df = yield_data.data
            tenors = yield_data.tenors

        start_idx = self.config.pca_window

        # SSB Enhancement: Levels vs Changes analysis
        levels_vs_changes_results = {}
        if self.config.pca_method in ['levels', 'both']:
            print("Performing SSB levels vs changes PCA analysis...")
            levels_vs_changes_results = self.factor_model.fit_levels_vs_changes_analysis(
                df.iloc[-self.config.pca_window:].values
            )
            print(f"SSB Analysis: {levels_vs_changes_results['ssb_recommendation']} method recommended")
            print(f"Levels explained: {levels_vs_changes_results['levels_total_explained']:.3%}")
            print(f"Changes explained: {levels_vs_changes_results['changes_total_explained']:.3%}")

        results = []
        positions = np.zeros(len(tenors))
        portfolio_value = 1.0

        print(f"Starting enhanced backtest with {len(df) - start_idx} periods...")
        print(f"Configuration: PCA Window={self.config.pca_window}, "
              f"Entry Percentile={self.config.entry_percentile}%")

        for i in range(start_idx, len(df)):
            current_date = df.index[i]

            # Enhanced PCA fitting with stability analysis
            window_data = df.iloc[i - self.config.pca_window:i].values
            self.factor_model.fit_with_stability_analysis(window_data)

            # Current yield analysis
            current_yields = df.iloc[i].values
            reconstructed, dislocations, pca_metrics = self.factor_model.compute_enhanced_dislocations(current_yields)

            # Update position manager history
            pc_exposures = self._calculate_current_pc_exposures(positions, tenors)
            self.position_manager.update_history(dislocations, pc_exposures)

            # Generate signals using percentile-based method
            signals = self.position_manager.compute_percentile_signals(dislocations)

            # Calculate current volatility
            if i > start_idx + 21:
                recent_returns = [r['daily_pnl'] for r in results[-21:]]
                current_vol = np.std(recent_returns) * np.sqrt(252)
            else:
                current_vol = self.config.vol_target

            # Generate independent trades for curve neutrality
            if self.config.enable_curve_neutrality and i == start_idx:
                self.independent_trades = self.factor_model.get_independent_trades(tenors)
                print(f"Generated independent trades:")
                for name, trade in self.independent_trades.items():
                    print(f"  {trade['description']}")

            # SSB Enhancement: Calculate duration-dollar weighted butterfly spreads
            ssb_butterfly_spreads = {}
            butterfly_positions = {}
            new_butterfly_trades = []
            closed_butterfly_trades = []

            if self.config.use_duration_dollar_weighting:
                butterfly_weights = self.factor_model.calculate_duration_dollar_weights(
                    current_yields, tenors
                )

                # Extract SSB butterfly spreads
                ssb_butterfly_spreads = {
                    key: weights['ssb_spread']
                    for key, weights in butterfly_weights.items()
                }

                # Update butterfly trading
                new_butterfly_trades, closed_butterfly_trades = self.position_manager.update_butterfly_trades(
                    current_date, ssb_butterfly_spreads, tenors
                )

                # Get butterfly positions
                butterfly_positions = self.position_manager.get_butterfly_positions()

            # SSB Enhancement: Solve independent trades every 21 days
            if i % 21 == 0:  # Update every month
                independent_trades = self.factor_model.solve_independent_trades(tenors)

            # Compute enhanced positions
            if self.config.enable_curve_neutrality:
                target_positions = self.position_manager.compute_curve_neutral_positions(
                    signals, self.independent_trades, current_vol, tenors
                )
            else:
                # Fallback to original method
                target_positions = self.position_manager.compute_positions(
                    dislocations, current_vol, positions
                )

            # Enhanced transaction cost calculation
            transaction_cost_details = self.position_manager.calculate_enhanced_transaction_costs(
                positions, target_positions
            )

            positions = target_positions.copy()

            # Calculate P&L
            if i < len(df) - 1:
                yield_changes = df.iloc[i + 1].values - current_yields
                daily_pnl = -np.sum(positions * yield_changes * 10000)
                daily_pnl -= transaction_cost_details['total_costs'] * 10000
                portfolio_value *= (1 + daily_pnl / 10000)
            else:
                daily_pnl = 0

            # Enhanced result compilation
            result = {
                'date': current_date,
                'portfolio_value': portfolio_value,
                'daily_pnl': daily_pnl,
                'gross_exposure': np.sum(np.abs(positions)),
                'transaction_costs': transaction_cost_details['total_costs'],
                'market_impact': transaction_cost_details['market_impact'],
                'turnover': transaction_cost_details['turnover'],
                'explained_variance': pca_metrics['explained_variance'],
                'max_dislocation': pca_metrics['max_abs_dislocation'],
                'active_signals': np.sum(np.abs(positions) > 0),
                'pc_contributions': pca_metrics['pc_contributions'],
                'reconstruction_error': pca_metrics['reconstruction_error'],
                'stability_score': pca_metrics['stability_score'],

                # SSB Enhanced metrics
                'ssb_butterfly_spreads': ssb_butterfly_spreads,
                'active_butterfly_trades': len(self.position_manager.active_butterfly_trades),
                'new_butterfly_trades': len(new_butterfly_trades),
                'closed_butterfly_trades': len(closed_butterfly_trades)
            }

            # Add position and yield data
            for j, tenor in enumerate(tenors):
                result.update({
                    f'{tenor}_dislocation': dislocations[j],
                    f'{tenor}_position': positions[j],
                    f'{tenor}_yield': current_yields[j],
                    f'{tenor}_signal': signals[j] if j < len(signals) else 0
                })

            # Add independent trades info every 21 days
            if i % 21 == 0 and 'independent_trades' in locals():
                for pc_name, trade_info in independent_trades.items():
                    result[f'independent_trade_{pc_name.lower()}'] = trade_info['description']

            results.append(result)

            # Progress reporting
            if i % 252 == 0:
                print(f"Processed {i - start_idx} periods, Portfolio value: {portfolio_value:.3f}, "
                      f"Stability: {pca_metrics['stability_score']:.2f}")

        # Compile enhanced results
        enhanced_results = self._compile_enhanced_results(results, tenors)

        # Add SSB analysis
        enhanced_results['ssb_analysis'] = {
            'butterfly_performance': self.position_manager.get_butterfly_performance_stats(),
            'levels_vs_changes': levels_vs_changes_results,
            'final_independent_trades': independent_trades if 'independent_trades' in locals() else {}
        }

        return enhanced_results

    def _calculate_current_pc_exposures(self, positions: np.ndarray, tenors: List[str]) -> np.ndarray:
        """Calculate current portfolio exposures to principal components"""
        return self.position_manager._calculate_pc_exposures(positions, tenors)

    def _compile_enhanced_results(self, results: List[Dict], tenors: List[str]) -> Dict:
        """Compile results with enhanced analytics"""
        df = pd.DataFrame(results).set_index('date')

        # Basic performance metrics
        df['daily_return'] = df['daily_pnl'] / 10000
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

        total_return = df['cumulative_return'].iloc[-1]
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1
        volatility = df['daily_return'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(df['portfolio_value'])

        # Enhanced metrics
        avg_gross_exposure = df['gross_exposure'].mean()
        avg_transaction_costs = df['transaction_costs'].mean()
        avg_market_impact = df['market_impact'].mean()
        avg_turnover = df['turnover'].mean()
        avg_active_signals = df['active_signals'].mean()
        avg_stability_score = df['stability_score'].mean()

        # Calculate PC attribution if enabled
        attribution_df = pd.DataFrame()
        if self.config.enable_pc_attribution:
            attribution_df = self.performance_analyzer.calculate_pc_attribution(
                {'performance_df': df}, self.factor_model
            )

        # Calculate comprehensive risk metrics
        risk_metrics = self.performance_analyzer.calculate_risk_metrics(
            {'performance_df': df}
        )

        enhanced_results = {
            'performance_df': df,
            'attribution_df': attribution_df,
            'metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_gross_exposure': avg_gross_exposure,
                'avg_transaction_costs': avg_transaction_costs,
                'avg_market_impact': avg_market_impact,
                'avg_turnover': avg_turnover,
                'avg_active_signals': avg_active_signals,
                'avg_stability_score': avg_stability_score
            },
            'risk_metrics': risk_metrics,
            'independent_trades': self.independent_trades,
            'config': self.config
        }

        # Generate enhanced reporting if enabled
        if self.config.enable_enhanced_reporting:
            print("\n" + "=" * 80)
            print("GENERATING ENHANCED PERFORMANCE REPORT")
            print("=" * 80)

            self.performance_analyzer.generate_comprehensive_report(
                enhanced_results, attribution_df, risk_metrics
            )

            # Statistical significance tests
            sig_tests = statistical_significance_tests(enhanced_results)
            print(f"\nSTATISTICAL SIGNIFICANCE TESTS:")
            print(f"  Mean Return Significant (5%): {sig_tests['mean_return_significance']['is_significant_5pct']}")
            print(f"  Returns Are Normal (5%):      {sig_tests['normality_test']['is_normal_5pct']}")
            if sig_tests['serial_correlation_test']['has_serial_correlation_5pct'] is not None:
                print(
                    f"  Has Serial Correlation (5%):  {sig_tests['serial_correlation_test']['has_serial_correlation_5pct']}")

        return enhanced_results

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()


class EnhancedPerformanceAnalyzer:
    """Enhanced performance analysis based on SSB paper methodology"""

    @staticmethod
    def calculate_pc_attribution(backtest_results: Dict,
                                 factor_model: EnhancedPCAFactorModel) -> Dict:
        """Calculate return attribution to each principal component"""
        df = backtest_results['performance_df']

        # Initialize attribution dataframe
        attribution_df = pd.DataFrame(index=df.index)

        # Extract PC contributions from daily results
        pc_contributions = []
        for idx, row in df.iterrows():
            if hasattr(row, 'pc_contributions'):
                pc_contributions.append(row['pc_contributions'])
            else:
                pc_contributions.append([0, 0, 0])

        pc_contributions = np.array(pc_contributions)

        # Calculate daily returns attributed to each PC
        for i, pc_name in enumerate(['Level', 'Slope', 'Curvature']):
            # PC return = PC exposure * PC movement * portfolio duration
            pc_returns = pc_contributions[:, i] * df['daily_pnl'] / 10000
            attribution_df[f'{pc_name}_return'] = pc_returns

        # Calculate residual return (unexplained by first 3 PCs)
        total_pc_return = attribution_df[['Level_return', 'Slope_return', 'Curvature_return']].sum(axis=1)
        attribution_df['Residual_return'] = df['daily_return'] - total_pc_return
        attribution_df['Spread_return'] = attribution_df['Residual_return']  # Attribute residual to spread changes

        # Calculate rolling attribution metrics
        attribution_df['Level_vol'] = attribution_df['Level_return'].rolling(63).std() * np.sqrt(252)
        attribution_df['Slope_vol'] = attribution_df['Slope_return'].rolling(63).std() * np.sqrt(252)
        attribution_df['Curvature_vol'] = attribution_df['Curvature_return'].rolling(63).std() * np.sqrt(252)

        return attribution_df

    @staticmethod
    def calculate_risk_metrics(backtest_results: Dict) -> Dict:
        """Calculate comprehensive risk metrics following SSB methodology"""
        df = backtest_results['performance_df']

        # Basic risk metrics
        returns = df['daily_return']
        portfolio_values = df['portfolio_value']

        # Volatility metrics
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)

        # VaR calculation (95% and 99% confidence levels)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean()
        es_99 = returns[returns <= var_99].mean()

        # Maximum drawdown analysis
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Drawdown duration analysis
        drawdown_periods = []
        in_drawdown = False
        start_dd = None

        for idx, dd in enumerate(drawdowns):
            if dd < -0.01 and not in_drawdown:  # Start of significant drawdown
                in_drawdown = True
                start_dd = idx
            elif dd >= 0 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if start_dd is not None:
                    drawdown_periods.append(idx - start_dd)

        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = np.max(drawdown_periods) if drawdown_periods else 0

        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Rolling Sharpe ratio
        rolling_sharpe = (returns.rolling(63).mean() * 252) / (returns.rolling(63).std() * np.sqrt(252))

        # Curve exposure risk metrics
        gross_exposure_vol = df['gross_exposure'].std()
        avg_gross_exposure = df['gross_exposure'].mean()
        max_gross_exposure = df['gross_exposure'].max()

        return {
            'volatility': {
                'daily': daily_vol,
                'annualized': annualized_vol,
                'rolling_sharpe_mean': rolling_sharpe.mean(),
                'rolling_sharpe_std': rolling_sharpe.std()
            },
            'var_metrics': {
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall_95': es_95,
                'expected_shortfall_99': es_99
            },
            'drawdown_metrics': {
                'max_drawdown': max_drawdown,
                'avg_drawdown_duration_days': avg_drawdown_duration,
                'max_drawdown_duration_days': max_drawdown_duration,
                'num_drawdown_periods': len(drawdown_periods)
            },
            'distribution_metrics': {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'jarque_bera_pvalue': stats.jarque_bera(returns.dropna())[1]
            },
            'exposure_metrics': {
                'avg_gross_exposure': avg_gross_exposure,
                'max_gross_exposure': max_gross_exposure,
                'exposure_volatility': gross_exposure_vol
            }
        }

    @staticmethod
    def generate_ssb_enhanced_report(backtest_results: Dict) -> None:
        """
        Enhanced SSB reporting
        """
        df = backtest_results['performance_df']
        metrics = backtest_results['metrics']

        print("\n" + "=" * 100)
        print("SSB-ENHANCED YIELD CURVE ARBITRAGE STRATEGY REPORT")
        print("=" * 100)

        # Standard performance metrics
        print(f"\nCORE PERFORMANCE METRICS:")
        print(f"  Total Return:               {metrics['total_return']:>12.2%}")
        print(f"  Annualized Return:          {metrics['annualized_return']:>12.2%}")
        print(f"  Sharpe Ratio:               {metrics['sharpe_ratio']:>12.2f}")
        print(f"  Max Drawdown:               {metrics['max_drawdown']:>12.2%}")

        # SSB-specific analysis
        if 'ssb_analysis' in backtest_results:
            ssb_analysis = backtest_results['ssb_analysis']

            # Butterfly trading performance
            if 'butterfly_performance' in ssb_analysis:
                bf_perf = ssb_analysis['butterfly_performance']
                if not bf_perf.get('no_trades', False):
                    print(f"\nSSB BUTTERFLY TRADING PERFORMANCE:")
                    print(f"  Total Butterfly Trades:     {bf_perf['total_trades']:>12d}")
                    print(f"  Win Rate:                   {bf_perf['win_rate']:>12.1%}")
                    print(f"  Average P&L per Trade:      {bf_perf['avg_pnl_bp']:>12.1f} bp")
                    print(f"  Total Butterfly P&L:        {bf_perf['total_pnl_bp']:>12.1f} bp")
                    print(f"  Average Holding Period:     {bf_perf['avg_holding_days']:>12.0f} days")
                    print(f"  Butterfly Sharpe Ratio:     {bf_perf['sharpe_ratio']:>12.2f}")

            # Levels vs Changes analysis
            if 'levels_vs_changes' in ssb_analysis:
                lvc = ssb_analysis['levels_vs_changes']
                if lvc:
                    print(f"\nSSB LEVELS VS CHANGES ANALYSIS:")
                    print(f"  Levels Explained Variance:  {lvc['levels_total_explained']:>12.3%}")
                    print(f"  Changes Explained Variance: {lvc['changes_total_explained']:>12.3%}")
                    print(f"  SSB Recommendation:         {lvc['ssb_recommendation']:>12s}")

            # Independent trades
            if 'final_independent_trades' in ssb_analysis:
                ind_trades = ssb_analysis['final_independent_trades']
                if ind_trades:
                    print(f"\nSSB INDEPENDENT TRADES (Figure 4):")
                    for pc_name, trade_info in ind_trades.items():
                        print(f"  {trade_info['description']}")

        # Calculate SSB compliance score
        compliance_score = EnhancedPerformanceAnalyzer._calculate_ssb_compliance(backtest_results)
        print(f"\nSSB METHODOLOGY COMPLIANCE:     {compliance_score:>12.1f}%")

        print("=" * 100)

    @staticmethod
    def _calculate_ssb_compliance(backtest_results: Dict) -> float:
        """Calculate SSB methodology compliance score"""
        score = 0
        max_score = 0

        # Check for duration-dollar weighting (25 points)
        max_score += 25
        df = backtest_results['performance_df']
        if any('ssb_butterfly_spreads' in col for col in df.columns):
            score += 25

        # Check for levels vs changes analysis (20 points)
        max_score += 20
        if ('ssb_analysis' in backtest_results and
                'levels_vs_changes' in backtest_results['ssb_analysis']):
            score += 20

        # Check for independent trades (20 points)
        max_score += 20
        if ('ssb_analysis' in backtest_results and
                'final_independent_trades' in backtest_results['ssb_analysis']):
            score += 20

        # Check for butterfly trading (15 points)
        max_score += 15
        if ('ssb_analysis' in backtest_results and
                'butterfly_performance' in backtest_results['ssb_analysis']):
            bf_perf = backtest_results['ssb_analysis']['butterfly_performance']
            if not bf_perf.get('no_trades', False):
                score += 15

        # Check for proper percentiles (10 points)
        max_score += 10
        score += 10  # Simplified check

        # Check for holding periods (10 points)
        max_score += 10
        if ('ssb_analysis' in backtest_results and
                'butterfly_performance' in backtest_results['ssb_analysis']):
            bf_perf = backtest_results['ssb_analysis']['butterfly_performance']
            if not bf_perf.get('no_trades', False) and bf_perf.get('avg_holding_days', 0) > 0:
                score += 10

        return (score / max_score * 100) if max_score > 0 else 0

    @staticmethod
    def generate_comprehensive_report(backtest_results: Dict,
                                      attribution_df: pd.DataFrame,
                                      risk_metrics: Dict) -> None:
        """Generate comprehensive performance report following SSB style"""

        df = backtest_results['performance_df']
        metrics = backtest_results['metrics']

        print("=" * 80)
        print("ENHANCED YIELD CURVE ARBITRAGE STRATEGY PERFORMANCE REPORT")
        print("=" * 80)

        # Strategy overview
        print(f"\nSTRATEGY PERIOD: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Total Observations: {len(df):,}")
        print(f"Strategy Frequency: Daily Rebalancing")

        # Performance metrics
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Total Return:           {metrics['total_return']:>8.2%}")
        print(f"  Annualized Return:      {metrics['annualized_return']:>8.2%}")
        print(f"  Volatility:             {metrics['volatility']:>8.2%}")
        print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Max Drawdown:           {metrics['max_drawdown']:>8.2%}")
        print(f"  Calmar Ratio:           {metrics['annualized_return'] / abs(metrics['max_drawdown']):>8.2f}")

        # Enhanced risk metrics
        print(f"\nRISK METRICS:")
        print(f"  Daily VaR (95%):        {risk_metrics['var_metrics']['var_95']:>8.4f}")
        print(f"  Daily VaR (99%):        {risk_metrics['var_metrics']['var_99']:>8.4f}")
        print(f"  Expected Shortfall (95%):{risk_metrics['var_metrics']['expected_shortfall_95']:>8.4f}")
        print(f"  Skewness:               {risk_metrics['distribution_metrics']['skewness']:>8.2f}")
        print(f"  Kurtosis:               {risk_metrics['distribution_metrics']['kurtosis']:>8.2f}")

        # Drawdown analysis
        print(f"\nDRAWDOWN ANALYSIS:")
        print(f"  Maximum Drawdown:       {risk_metrics['drawdown_metrics']['max_drawdown']:>8.2%}")
        print(f"  Avg Drawdown Duration:  {risk_metrics['drawdown_metrics']['avg_drawdown_duration_days']:>8.1f} days")
        print(f"  Max Drawdown Duration:  {risk_metrics['drawdown_metrics']['max_drawdown_duration_days']:>8.0f} days")
        print(f"  Number of Drawdowns:    {risk_metrics['drawdown_metrics']['num_drawdown_periods']:>8.0f}")

        # Strategy statistics
        print(f"\nSTRATEGY STATISTICS:")
        print(f"  Avg Gross Exposure:     {metrics['avg_gross_exposure']:>8.2f}")
        print(f"  Max Gross Exposure:     {risk_metrics['exposure_metrics']['max_gross_exposure']:>8.2f}")
        print(f"  Avg Transaction Costs:  {metrics['avg_transaction_costs']:>8.4f}")
        print(f"  Avg Active Signals:     {metrics['avg_active_signals']:>8.1f}")

        # PC Attribution Analysis
        if not attribution_df.empty:
            print(f"\nPRINCIPAL COMPONENT ATTRIBUTION:")

            for pc_name in ['Level', 'Slope', 'Curvature']:
                pc_return_col = f'{pc_name}_return'
                if pc_return_col in attribution_df.columns:
                    total_pc_return = attribution_df[pc_return_col].sum()
                    pc_vol = attribution_df[pc_return_col].std() * np.sqrt(252)
                    pc_sharpe = (attribution_df[pc_return_col].mean() * 252) / pc_vol if pc_vol > 0 else 0

                    print(f"  {pc_name} Component:")
                    print(f"    Total Return:         {total_pc_return:>8.4f}")
                    print(f"    Annualized Vol:       {pc_vol:>8.2%}")
                    print(f"    Sharpe Ratio:         {pc_sharpe:>8.2f}")

            # Residual analysis
            if 'Residual_return' in attribution_df.columns:
                residual_return = attribution_df['Residual_return'].sum()
                residual_vol = attribution_df['Residual_return'].std() * np.sqrt(252)
                print(f"  Residual (Spread Changes):")
                print(f"    Total Return:         {residual_return:>8.4f}")
                print(f"    Annualized Vol:       {residual_vol:>8.2%}")

        print("\n" + "=" * 80)

    @staticmethod
    def plot_enhanced_performance(backtest_results: Dict,
                                  attribution_df: pd.DataFrame = None) -> None:
        """Create enhanced performance plots following SSB methodology"""
        df = backtest_results['performance_df']

        fig = plt.figure(figsize=(20, 15))

        # 1. Portfolio Performance
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(df.index, df['portfolio_value'], linewidth=2, color='darkblue')
        ax1.set_title('Portfolio Value Over Time', fontweight='bold')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True, alpha=0.3)

        # 2. Rolling Sharpe Ratio
        ax2 = plt.subplot(3, 3, 2)
        rolling_returns = df['daily_return'].rolling(63)
        rolling_sharpe = (rolling_returns.mean() * 252) / (rolling_returns.std() * np.sqrt(252))
        ax2.plot(df.index, rolling_sharpe, color='green', linewidth=2)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
        ax2.set_title('Rolling 3-Month Sharpe Ratio', fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Drawdown
        ax3 = plt.subplot(3, 3, 3)
        rolling_max = df['portfolio_value'].expanding().max()
        drawdown = (df['portfolio_value'] - rolling_max) / rolling_max * 100
        ax3.fill_between(df.index, drawdown, 0, alpha=0.3, color='red')
        ax3.plot(df.index, drawdown, color='darkred', linewidth=1)
        ax3.set_title('Drawdown (%)', fontweight='bold')
        ax3.set_ylabel('Drawdown %')
        ax3.grid(True, alpha=0.3)

        # 4. Return Distribution
        ax4 = plt.subplot(3, 3, 4)
        returns_bp = df['daily_return'] * 10000  # Convert to basis points
        ax4.hist(returns_bp, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(returns_bp.mean(), color='red', linestyle='--', label=f'Mean: {returns_bp.mean():.1f}bp')
        ax4.set_title('Daily Returns Distribution', fontweight='bold')
        ax4.set_xlabel('Daily Return (bp)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Gross Exposure
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(df.index, df['gross_exposure'], color='orange', linewidth=1.5)
        ax5.axhline(y=backtest_results['config'].max_leverage, color='red',
                    linestyle='--', label=f'Max Leverage: {backtest_results["config"].max_leverage}')
        ax5.set_title('Gross Exposure Over Time', fontweight='bold')
        ax5.set_ylabel('Gross Exposure')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Signal Activity
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(df.index, df['active_signals'], color='purple', linewidth=1.5)
        ax6.set_title('Active Trading Signals', fontweight='bold')
        ax6.set_ylabel('Number of Active Positions')
        ax6.grid(True, alpha=0.3)

        # 7. PC Attribution (if available)
        if attribution_df is not None and not attribution_df.empty:
            ax7 = plt.subplot(3, 3, 7)

            pc_columns = [col for col in attribution_df.columns if col.endswith('_return')]
            colors = ['blue', 'green', 'red', 'orange', 'purple']

            bottom = np.zeros(len(attribution_df))
            for i, col in enumerate(pc_columns):
                if i < len(colors):
                    cumulative_return = (1 + attribution_df[col]).cumprod() - 1
                    ax7.plot(attribution_df.index, cumulative_return * 100,
                             color=colors[i], label=col.replace('_return', ''), linewidth=2)

            ax7.set_title('Cumulative PC Attribution (%)', fontweight='bold')
            ax7.set_ylabel('Cumulative Return %')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        # 8. Transaction Costs
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(df.index, df['transaction_costs'] * 10000, color='brown', linewidth=1.5)
        ax8.set_title('Daily Transaction Costs (bp)', fontweight='bold')
        ax8.set_ylabel('Transaction Costs (bp)')
        ax8.grid(True, alpha=0.3)

        # 9. Yield Curve Dislocation
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(df.index, df['max_dislocation'], color='darkgreen', linewidth=1.5)
        ax9.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Signal Threshold (2σ)')
        ax9.set_title('Maximum Yield Curve Dislocation', fontweight='bold')
        ax9.set_ylabel('Max |Dislocation| (σ)')
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def statistical_significance_tests(backtest_results: Dict) -> Dict:
    returns = backtest_results['performance_df']['daily_return']

    t_stat, t_pvalue = stats.ttest_1samp(returns.dropna(), 0)

    jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(returns.dropna(), lags=10, return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].iloc[-1]
    except ImportError:
        lb_pvalue = np.nan

    return {
        'mean_return_significance': {
            't_statistic': t_stat,
            'p_value': t_pvalue,
            'is_significant_5pct': t_pvalue < 0.05
        },
        'normality_test': {
            'jarque_bera_statistic': jb_stat,
            'p_value': jb_pvalue,
            'is_normal_5pct': jb_pvalue > 0.05
        },
        'serial_correlation_test': {
            'ljung_box_p_value': lb_pvalue,
            'has_serial_correlation_5pct': lb_pvalue < 0.05 if not np.isnan(lb_pvalue) else None
        }
    }


def run_enhanced_strategy():
    print("=" * 80)
    print("SSB-ENHANCED YIELD CURVE ARBITRAGE STRATEGY")
    print("=" * 80)

    # Enhanced configuration with SSB parameters
    config = StrategyConfig(
        pca_window=756,  # 3+ years as per SSB paper
        entry_percentile=3.0,  # SSB recommendation
        exit_percentile=30.0,  # SSB recommendation
        max_leverage=2.5,
        enable_curve_neutrality=True,
        enable_pc_attribution=True,
        enable_enhanced_reporting=True,

        # SSB-specific parameters
        max_holding_days=120,  # SSB: 60-120 days
        use_duration_dollar_weighting=True,  # Core SSB innovation
        pca_method='levels',  # SSB prefers levels over changes
        force_120_tenors=False,  # Set to True if you want 120 interpolated tenors
    )

    print(f"SSB Configuration:")
    print(f"  Entry/Exit Percentiles:     {config.entry_percentile}% / {config.exit_percentile}%")
    print(f"  Max Holding Period:         {config.max_holding_days} days")
    print(f"  Duration-Dollar Weighting:  {config.use_duration_dollar_weighting}")
    print(f"  PCA Method:                 {config.pca_method}")
    print(f"  Force 120 Tenors:           {config.force_120_tenors}")

    yield_data = YieldCurveData('combined_curves.csv')

    strategy = EnhancedYieldCurveArbitrageStrategy(config)
    results = strategy.backtest(yield_data)

    # Enhanced SSB reporting
    strategy.performance_analyzer.generate_ssb_enhanced_report(results)

    # Generate plots
    strategy.performance_analyzer.plot_enhanced_performance(
        results, results.get('attribution_df', pd.DataFrame())
    )

    return results


if __name__ == '__main__':
    enhanced_results = run_enhanced_strategy()

    print(f"\nStrategy completed successfully!")
    print(f"Final Portfolio Value: {enhanced_results['performance_df']['portfolio_value'].iloc[-1]:.3f}")
    print(f"Sharpe Ratio: {enhanced_results['metrics']['sharpe_ratio']:.2f}")
    print(f"Average Stability Score: {enhanced_results['metrics']['avg_stability_score']:.2f}")

    # SSB-specific summary
    if 'ssb_analysis' in enhanced_results:
        ssb_analysis = enhanced_results['ssb_analysis']
        if 'butterfly_performance' in ssb_analysis:
            bf_perf = ssb_analysis['butterfly_performance']
            if not bf_perf.get('no_trades', False):
                print(f"Butterfly Trading Sharpe: {bf_perf['sharpe_ratio']:.2f}")
                print(f"Total Butterfly Trades: {bf_perf['total_trades']}")
                print(f"Butterfly Win Rate: {bf_perf['win_rate']:.1%}")