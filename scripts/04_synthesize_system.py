#!/usr/bin/env python3
"""Pipeline Step 4: Synthesize a unified trading system document from the knowledge base.

Usage:
    python scripts/04_synthesize_system.py
    python scripts/04_synthesize_system.py --knowledge data/knowledge/knowledge_base.json
    python scripts/04_synthesize_system.py --output data/knowledge/trading_system.md
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
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from config.settings import Settings
from src.analysis.gemini_client import GeminiClient, RateLimiter
from src.knowledge.schema import TradingKnowledgeBase
from src.knowledge.synthesizer import TradingSynthesizer

console = Console()

DEFAULT_KNOWLEDGE = "data/knowledge/knowledge_base.json"
DEFAULT_OUTPUT = "data/knowledge/trading_system.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize a cohesive trading system document from the knowledge base.",
    )
    parser.add_argument(
        "--knowledge",
        type=str,
        default=DEFAULT_KNOWLEDGE,
        help=f"Path to the knowledge base JSON (default: {DEFAULT_KNOWLEDGE}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output markdown file path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Use rule-based synthesis only (no Gemini API call).",
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


async def main(args: argparse.Namespace) -> None:
    settings = Settings()
    knowledge_path = Path(args.knowledge)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kb = _load_knowledge_base(knowledge_path)

    console.print(
        Panel(
            f"[bold cyan]Step 4: Synthesize Trading System[/bold cyan]\n"
            f"Knowledge base: [green]{knowledge_path}[/green]\n"
            f"Strategies:     [green]{len(kb.strategies)}[/green]\n"
            f"Indicators:     [green]{len(kb.indicators)}[/green]\n"
            f"Output:         [green]{output_path}[/green]\n"
            f"AI mode:        [green]{'off' if args.no_ai else 'on'}[/green]",
            title="Trading Bot Pipeline",
            border_style="blue",
        )
    )

    if not kb.strategies:
        console.print(
            "[yellow]Warning: No strategies found in knowledge base. "
            "The output document will be minimal.[/yellow]"
        )

    # Build synthesizer with optional AI client
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
            console.print("[yellow]Gemini not available, falling back to rule-based synthesis.[/yellow]")

    try:
        synthesizer = TradingSynthesizer(gemini_client=gemini_client)

        with console.status("[bold green]Generating trading system document..."):
            document = await synthesizer.synthesize(kb)
    except Exception as exc:
        console.print(f"[red]Error during synthesis:[/red] {exc}")
        sys.exit(1)

    try:
        output_path.write_text(document, encoding="utf-8")
    except Exception as exc:
        console.print(f"[red]Error saving document:[/red] {exc}")
        sys.exit(1)

    console.print()
    table = Table(title="Synthesis Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Strategies Included", str(len(kb.strategies)))
    table.add_row("Indicators Referenced", str(len(kb.indicators)))
    table.add_row("General Rules", str(len(kb.general_rules)))
    table.add_row("Document Size", f"{len(document):,} characters")
    table.add_row("Document Lines", str(document.count('\n') + 1))
    table.add_row("Output File", str(output_path))
    console.print(table)

    if kb.strategies:
        console.print()
        console.print("[bold]Top Strategies by Confidence:[/bold]")
        sorted_strategies = sorted(
            kb.strategies, key=lambda s: s.confidence_score, reverse=True
        )
        for i, strategy in enumerate(sorted_strategies[:5], 1):
            score_bar = int(strategy.confidence_score * 10) * "#"
            console.print(
                f"  {i}. [cyan]{strategy.name}[/cyan] "
                f"[dim](confidence={strategy.confidence_score:.2f} [{score_bar:<10}], "
                f"quantifiability={strategy.quantifiability_score:.2f})[/dim]"
            )

    preview_lines = document.split("\n")[:20]
    preview = "\n".join(preview_lines)
    if len(document.split("\n")) > 20:
        preview += "\n..."
    console.print()
    console.print(
        Panel(
            Markdown(preview),
            title="Document Preview (first 20 lines)",
            border_style="dim",
        )
    )

    console.print(
        f"\n[bold green]Done![/bold green] Trading system document saved to "
        f"[cyan]{output_path}[/cyan]"
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
