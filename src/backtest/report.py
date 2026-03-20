"""Performance reporting for backtest results.

Generates markdown reports and rich console summaries from one or more
``BacktestResult`` objects.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.backtest.runner import BacktestResult

logger = logging.getLogger(__name__)

# Default output directory for reports.
_DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "data" / "backtest_results"


class ReportGenerator:
    """Build markdown reports and console summaries from backtest results."""

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        results: list[BacktestResult],
        output_path: Path | None = None,
    ) -> str:
        """Generate a markdown report from one or more backtest results.

        Parameters
        ----------
        results:
            List of ``BacktestResult`` objects.
        output_path:
            File to write the report to.  If ``None`` a default path
            under ``data/backtest_results/`` is used.

        Returns
        -------
        str
            The markdown text of the report.
        """
        if not results:
            return "# Backtest Report\n\nNo results to report.\n"

        if output_path is None:
            _DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = _DEFAULT_REPORT_DIR / f"report_{timestamp}.md"

        lines: list[str] = []
        lines.append("# Backtest Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary statistics across all results
        if len(results) > 1:
            lines.append("## Comparison Table")
            lines.append("")
            lines.append(self._comparison_table(results))
            lines.append("")

        # Individual result sections
        for idx, result in enumerate(results, 1):
            lines.append(f"## {idx}. {result.strategy_name}")
            lines.append("")
            lines.append(self._format_result(result))
            lines.append("")

        # Aggregate summary
        if len(results) > 1:
            lines.append("## Summary Statistics")
            lines.append("")
            lines.append(self._summary_statistics(results))
            lines.append("")

        report = "\n".join(lines)

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return report

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_result(self, result: BacktestResult) -> str:
        """Format a single BacktestResult as a markdown section."""
        lines: list[str] = []
        lines.append(f"- **Symbol:** {result.symbol}")
        lines.append(f"- **Period:** {result.period}")
        lines.append(f"- **Initial Cash:** ${result.initial_cash:,.2f}")
        lines.append(f"- **Final Value:** ${result.final_value:,.2f}")
        lines.append(f"- **Total Return:** {result.total_return_pct:+.2f}%")
        lines.append(
            f"- **Sharpe Ratio:** "
            f"{result.sharpe_ratio:.4f}" if result.sharpe_ratio is not None else "- **Sharpe Ratio:** N/A"
        )
        lines.append(
            f"- **Max Drawdown:** "
            f"{result.max_drawdown_pct:.2f}%" if result.max_drawdown_pct is not None else "- **Max Drawdown:** N/A"
        )
        lines.append(f"- **Total Trades:** {result.total_trades}")
        lines.append(
            f"- **Win Rate:** "
            f"{result.win_rate * 100:.1f}%" if result.win_rate is not None else "- **Win Rate:** N/A"
        )
        lines.append(
            f"- **Profit Factor:** "
            f"{result.profit_factor:.4f}" if result.profit_factor is not None else "- **Profit Factor:** N/A"
        )
        return "\n".join(lines)

    @staticmethod
    def _comparison_table(results: list[BacktestResult]) -> str:
        """Build a markdown comparison table for multiple results."""
        header = "| Strategy | Symbol | Return (%) | Sharpe | Max DD (%) | Trades | Win Rate (%) | Profit Factor |"
        separator = "|----------|--------|-----------|--------|-----------|--------|-------------|--------------|"
        rows: list[str] = []
        for r in results:
            sharpe = f"{r.sharpe_ratio:.2f}" if r.sharpe_ratio is not None else "N/A"
            dd = f"{r.max_drawdown_pct:.2f}" if r.max_drawdown_pct is not None else "N/A"
            wr = f"{r.win_rate * 100:.1f}" if r.win_rate is not None else "N/A"
            pf = f"{r.profit_factor:.2f}" if r.profit_factor is not None else "N/A"
            rows.append(
                f"| {r.strategy_name} | {r.symbol} | {r.total_return_pct:+.2f} "
                f"| {sharpe} | {dd} | {r.total_trades} | {wr} | {pf} |"
            )
        return "\n".join([header, separator] + rows)

    @staticmethod
    def _summary_statistics(results: list[BacktestResult]) -> str:
        """Compute aggregate statistics over multiple results."""
        returns = [r.total_return_pct for r in results]
        avg_return = sum(returns) / len(returns)
        best = max(results, key=lambda r: r.total_return_pct)
        worst = min(results, key=lambda r: r.total_return_pct)

        sharpes = [r.sharpe_ratio for r in results if r.sharpe_ratio is not None]
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else None

        total_trades = sum(r.total_trades for r in results)

        lines: list[str] = [
            f"- **Strategies tested:** {len(results)}",
            f"- **Average return:** {avg_return:+.2f}%",
            f"- **Best performer:** {best.strategy_name} ({best.total_return_pct:+.2f}%)",
            f"- **Worst performer:** {worst.strategy_name} ({worst.total_return_pct:+.2f}%)",
        ]
        if avg_sharpe is not None:
            lines.append(f"- **Average Sharpe:** {avg_sharpe:.4f}")
        lines.append(f"- **Total trades across all strategies:** {total_trades}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_summary(self, result: BacktestResult) -> None:
        """Print a rich-formatted summary of a single backtest result.

        Parameters
        ----------
        result:
            The ``BacktestResult`` to display.
        """
        console = Console()

        table = Table(
            title=f"Backtest: {result.strategy_name}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Symbol", result.symbol)
        table.add_row("Period", result.period)
        table.add_row("Initial Cash", f"${result.initial_cash:,.2f}")
        table.add_row("Final Value", f"${result.final_value:,.2f}")

        # Colour-code the return
        ret_style = "green" if result.total_return_pct >= 0 else "red"
        table.add_row(
            "Total Return",
            f"[{ret_style}]{result.total_return_pct:+.2f}%[/{ret_style}]",
        )

        table.add_row(
            "Sharpe Ratio",
            f"{result.sharpe_ratio:.4f}" if result.sharpe_ratio is not None else "N/A",
        )

        dd_str = f"{result.max_drawdown_pct:.2f}%" if result.max_drawdown_pct is not None else "N/A"
        table.add_row("Max Drawdown", dd_str)

        table.add_row("Total Trades", str(result.total_trades))

        table.add_row(
            "Win Rate",
            f"{result.win_rate * 100:.1f}%" if result.win_rate is not None else "N/A",
        )

        table.add_row(
            "Profit Factor",
            f"{result.profit_factor:.4f}" if result.profit_factor is not None else "N/A",
        )

        console.print()
        console.print(table)
        console.print()
