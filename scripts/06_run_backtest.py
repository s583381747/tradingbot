#!/usr/bin/env python3
"""Pipeline Step 6: Run backtests on generated (or custom) trading strategies.

Usage:
    python scripts/06_run_backtest.py --strategy src/strategy/generated/my_strategy.py
    python scripts/06_run_backtest.py --strategy all --symbol SPY
    python scripts/06_run_backtest.py --strategy all --symbol AAPL --start 2021-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.backtest.runner import BacktestRunner, BacktestResult
from src.backtest.report import ReportGenerator

console = Console()

DEFAULT_STRATEGY = "all"
DEFAULT_SYMBOL = "SPY"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_CASH = 100_000
DEFAULT_GENERATED_DIR = "src/strategy/generated/"
DEFAULT_RESULTS_DIR = "data/backtest_results/"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backtests on trading strategies using backtrader.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        help=(
            'Path to a strategy .py file, or "all" to run every strategy '
            f"in {DEFAULT_GENERATED_DIR} (default: {DEFAULT_STRATEGY})."
        ),
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help=f"Ticker symbol for backtest data (default: {DEFAULT_SYMBOL}).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START,
        help=f"Backtest start date YYYY-MM-DD (default: {DEFAULT_START}).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=DEFAULT_END,
        help=f"Backtest end date YYYY-MM-DD (default: {DEFAULT_END}).",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=DEFAULT_CASH,
        help=f"Starting cash amount (default: {DEFAULT_CASH:,.0f}).",
    )
    return parser.parse_args(argv)


def _discover_strategies(generated_dir: Path) -> list[Path]:
    if not generated_dir.exists():
        console.print(
            f"[red]Error:[/red] Generated strategies directory not found: {generated_dir}"
        )
        console.print(
            "[dim]Run step 05 first: python scripts/05_generate_strategies.py[/dim]"
        )
        sys.exit(1)

    files = sorted(
        f
        for f in generated_dir.glob("*.py")
        if f.name != "__init__.py" and not f.name.startswith("_")
    )

    if not files:
        console.print(f"[yellow]No strategy files found in {generated_dir}[/yellow]")
        sys.exit(0)

    return files


def _resolve_strategy_paths(strategy_arg: str) -> list[Path]:
    if strategy_arg.lower() == "all":
        return _discover_strategies(Path(DEFAULT_GENERATED_DIR))

    path = Path(strategy_arg)
    if not path.exists():
        console.print(f"[red]Error:[/red] Strategy file not found: {path}")
        sys.exit(1)
    if path.suffix != ".py":
        console.print(f"[red]Error:[/red] Strategy file must be a .py file: {path}")
        sys.exit(1)

    return [path]


def main(args: argparse.Namespace) -> None:
    results_dir = Path(DEFAULT_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    strategy_paths = _resolve_strategy_paths(args.strategy)
    is_comparison = len(strategy_paths) > 1

    mode_str = f"Comparing {len(strategy_paths)} strategies" if is_comparison else strategy_paths[0].name
    console.print(
        Panel(
            f"[bold cyan]Step 6: Run Backtest[/bold cyan]\n"
            f"Mode:      [green]{mode_str}[/green]\n"
            f"Symbol:    [green]{args.symbol}[/green]\n"
            f"Period:    [green]{args.start} to {args.end}[/green]\n"
            f"Cash:      [green]${args.cash:,.2f}[/green]\n"
            f"Results:   [green]{results_dir}[/green]",
            title="Trading Bot Pipeline",
            border_style="blue",
        )
    )

    runner = BacktestRunner(initial_cash=args.cash)
    reporter = ReportGenerator()

    all_results: list[BacktestResult] = []

    for i, strategy_path in enumerate(strategy_paths, 1):
        strategy_name = strategy_path.stem
        prefix = f"[{i}/{len(strategy_paths)}]" if is_comparison else ""

        console.print(
            f"\n{prefix} [bold]Running backtest:[/bold] [cyan]{strategy_name}[/cyan]"
        )

        try:
            with console.status(f"[bold green]Backtesting {strategy_name}..."):
                result = runner.run(
                    strategy_code_path=str(strategy_path),
                    symbol=args.symbol,
                    start=args.start,
                    end=args.end,
                )

            all_results.append(result)
            reporter.print_summary(result)

        except Exception as exc:
            console.print(f"  [red]Error:[/red] {exc}")

    # Generate report
    if all_results:
        console.print()
        try:
            report_path = results_dir / "report.md"
            report_text = reporter.generate_report(
                results=all_results,
                output_path=report_path,
            )
            console.print(f"[dim]Report saved to {report_path}[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Warning: Could not generate report: {exc}[/yellow]")

    # Comparison table (if multiple strategies)
    if is_comparison and len(all_results) > 1:
        console.print()
        comp_table = Table(
            title=f"Strategy Comparison - {args.symbol} ({args.start} to {args.end})",
            show_header=True,
            header_style="bold magenta",
        )
        comp_table.add_column("Strategy", style="cyan")
        comp_table.add_column("Final Value", justify="right")
        comp_table.add_column("Return", justify="right")
        comp_table.add_column("Trades", justify="right")
        comp_table.add_column("Sharpe", justify="right")
        comp_table.add_column("Max DD", justify="right")

        sorted_results = sorted(all_results, key=lambda r: r.total_return_pct, reverse=True)

        for result in sorted_results:
            ret_color = "green" if result.total_return_pct >= 0 else "red"
            sharpe_str = f"{result.sharpe_ratio:.3f}" if result.sharpe_ratio is not None else "N/A"
            dd_str = f"{result.max_drawdown_pct:.2f}%" if result.max_drawdown_pct is not None else "N/A"
            comp_table.add_row(
                result.strategy_name,
                f"${result.final_value:,.2f}",
                f"[{ret_color}]{result.total_return_pct:+.2f}%[/]",
                str(result.total_trades),
                sharpe_str,
                dd_str,
            )

        console.print(comp_table)

    # Final summary
    console.print()
    if all_results:
        best = max(all_results, key=lambda r: r.total_return_pct)
        ret_color = "green" if best.total_return_pct >= 0 else "red"
        console.print(
            f"[bold]Best strategy:[/bold] [cyan]{best.strategy_name}[/cyan] "
            f"with [{ret_color}]{best.total_return_pct:+.2f}%[/] return"
        )

    failed_count = len(strategy_paths) - len(all_results)
    if failed_count:
        console.print(f"[yellow]{failed_count} strategy(ies) failed to backtest.[/yellow]")

    console.print(
        f"\n[bold green]Done![/bold green] Backtested {len(all_results)} strategy(ies) "
        f"on {args.symbol}."
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
