import numpy as np
import matplotlib.pyplot as plt

from config import BacktestResults
from metrics import compute_sharpe, compute_max_drawdown


def print_results(
        results: BacktestResults,
) -> None:
    pr = results.portfolio_returns
    br = results.benchmark_returns

    cum_bh = (1 + br).prod() - 1
    cum_strat = (1 + pr).prod() - 1
    sharpe = compute_sharpe(pr)
    max_dd = compute_max_drawdown(pr)
    win_rate = (pr > 0).sum() / len(pr)

    print("\n=== Overall Walk-Forward Performance ===")
    print(f"Equal-Weight Benchmark Return: {cum_bh:.2%}")
    print(f"Strategy Return: {cum_strat:.2%}")
    print(f"Strategy Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Number of periods: {len(pr)}")

    asset_returns = results.positions * results.actuals
    print("\n=== Per-Asset Contribution ===")
    stats = []
    for i, sym in enumerate(results.symbols):
        ar = asset_returns[:, i]
        stats.append((sym, compute_sharpe(ar), ar.sum()))

    for sym, sh, tot in sorted(stats, key=lambda x: -x[1]):
        print(f"  {sym}: Sharpe={sh:.2f}, Return Contribution={tot:.2%}")


def plot_results(
        results: BacktestResults,
) -> None:
    cum_strat = np.cumprod(1 + results.portfolio_returns) - 1
    cum_bench = np.cumprod(1 + results.benchmark_returns) - 1

    plt.figure(figsize=(16, 10))
    plt.plot(cum_strat, label="Strategy")
    plt.plot(cum_bench, label="Benchmark")
    plt.title("Strategy Returns")
    plt.legend()
    plt.ylabel("Return")

    plt.savefig("plots/strategy_returns.png", dpi=300)
    plt.show()