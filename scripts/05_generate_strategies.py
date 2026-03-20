#!/usr/bin/env python3
"""Pipeline Step 5: Generate executable strategy code from the knowledge base.

Usage:
    python scripts/05_generate_strategies.py
    python scripts/05_generate_strategies.py --top-n 3
    python scripts/05_generate_strategies.py --knowledge data/knowledge/knowledge_base.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from config.settings import Settings
from src.analysis.gemini_client import GeminiClient, RateLimiter
from src.knowledge.schema import TradingKnowledgeBase, TradingStrategy
from src.strategy.codegen import StrategyCodeGenerator

console = Console()

DEFAULT_KNOWLEDGE = "data/knowledge/knowledge_base.json"
DEFAULT_OUTPUT_DIR = "src/strategy/generated/"
DEFAULT_TOP_N = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate executable backtrader strategy code from the knowledge base.",
    )
    parser.add_argument(
        "--knowledge",
        type=str,
        default=DEFAULT_KNOWLEDGE,
        help=f"Path to the knowledge base JSON (default: {DEFAULT_KNOWLEDGE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated strategy files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top strategies to generate (default: {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Use template-based code generation only (no Gemini API call).",
    )
    return parser.parse_args(argv)


def _load_knowledge_base(path: Path) -> TradingKnowledgeBase:
    if not path.exists():
        console.print(f"[red]Error:[/red] Knowledge base not found: {path}")
        console.print("[dim]Run step 03 first: python scripts/03_extract_knowledge.py[/dim]")
        sys.exit(1)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TradingKnowledgeBase.model_validate(data)
    except Exception as exc:
        console.print(f"[red]Error loading knowledge base:[/red] {exc}")
        sys.exit(1)


def _rank_strategies(strategies: list[TradingStrategy]) -> list[TradingStrategy]:
    return sorted(
        strategies,
        key=lambda s: s.confidence_score * s.quantifiability_score,
        reverse=True,
    )


async def main(args: argparse.Namespace) -> None:
    settings = Settings()
    knowledge_path = Path(args.knowledge)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_n = args.top_n

    kb = _load_knowledge_base(knowledge_path)

    if not kb.strategies:
        console.print("[yellow]No strategies found in knowledge base. Nothing to generate.[/yellow]")
        sys.exit(0)

    ranked = _rank_strategies(kb.strategies)
    selected = ranked[:top_n]

    console.print(
        Panel(
            f"[bold cyan]Step 5: Generate Strategy Code[/bold cyan]\n"
            f"Knowledge base: [green]{knowledge_path}[/green]\n"
            f"Total strategies: [green]{len(kb.strategies)}[/green]\n"
            f"Generating top:   [green]{len(selected)}[/green]\n"
            f"Output dir:       [green]{output_dir}[/green]",
            title="Trading Bot Pipeline",
            border_style="blue",
        )
    )

    # Ranking table
    console.print()
    rank_table = Table(
        title="Strategy Ranking (confidence x quantifiability)",
        show_header=True,
        header_style="bold magenta",
    )
    rank_table.add_column("#", style="dim", width=4)
    rank_table.add_column("Strategy", style="cyan")
    rank_table.add_column("Confidence", justify="right")
    rank_table.add_column("Quantifiability", justify="right")
    rank_table.add_column("Score", justify="right", style="bold")
    rank_table.add_column("Selected", justify="center")

    for i, strategy in enumerate(ranked, 1):
        score = strategy.confidence_score * strategy.quantifiability_score
        is_selected = i <= top_n
        rank_table.add_row(
            str(i),
            strategy.name,
            f"{strategy.confidence_score:.2f}",
            f"{strategy.quantifiability_score:.2f}",
            f"{score:.3f}",
            "[green]Yes[/green]" if is_selected else "[dim]No[/dim]",
        )
    console.print(rank_table)

    # Build code generator with optional AI client
    gemini_client = None
    if not args.no_ai:
        try:
            rate_limiter = RateLimiter(
                requests_per_minute=settings.gemini.rpm,
                requests_per_day=settings.gemini.rpd,
            )
            gemini_client = GeminiClient(
                api_key=settings.gemini.api_key.get_secret_value(),
                model=settings.gemini.model,
                rate_limiter=rate_limiter,
            )
        except Exception:
            console.print("[yellow]Gemini not available, using template-based generation.[/yellow]")

    codegen = StrategyCodeGenerator(gemini_client=gemini_client)

    generated_files: list[Path] = []
    generation_errors: list[str] = []

    console.print()
    for i, strategy in enumerate(selected, 1):
        output_path = output_dir / f"{strategy.name_snake}.py"

        console.print(
            f"  [{i}/{len(selected)}] Generating [cyan]{strategy.name}[/cyan] "
            f"-> {output_path.name}...",
            end=" ",
        )

        try:
            code = await codegen.generate(strategy)
            output_path.write_text(code, encoding="utf-8")
            generated_files.append(output_path)
            console.print("[green]OK[/green]")
        except Exception as exc:
            generation_errors.append(f"{strategy.name}: {exc}")
            console.print(f"[red]FAILED ({exc})[/red]")

    # Summary
    console.print()
    summary_table = Table(title="Generation Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Strategies Available", str(len(kb.strategies)))
    summary_table.add_row("Strategies Selected", str(len(selected)))
    summary_table.add_row("Successfully Generated", str(len(generated_files)))
    summary_table.add_row(
        "Failed",
        f"[red]{len(generation_errors)}[/red]" if generation_errors else "0",
    )
    summary_table.add_row("Output Directory", str(output_dir))
    console.print(summary_table)

    if generated_files:
        console.print("\n[bold]Generated Files:[/bold]")
        for f in generated_files:
            size = f.stat().st_size
            console.print(f"  [green]+[/green] {f} ({size:,} bytes)")

        first_file = generated_files[0]
        preview_code = first_file.read_text(encoding="utf-8")
        preview_lines = preview_code.split("\n")[:30]
        preview = "\n".join(preview_lines)
        if len(preview_code.split("\n")) > 30:
            preview += "\n# ... (truncated)"

        console.print()
        console.print(
            Panel(
                Syntax(preview, "python", theme="monokai", line_numbers=True),
                title=f"Preview: {first_file.name}",
                border_style="dim",
            )
        )

    if generation_errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for err in generation_errors:
            console.print(f"  [red]-[/red] {err}")

    console.print(
        f"\n[bold green]Done![/bold green] Generated {len(generated_files)} strategy file(s) "
        f"in [cyan]{output_dir}[/cyan]"
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
