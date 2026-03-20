#!/usr/bin/env python3
"""Pipeline Step 3: Extract structured trading knowledge from video analyses.

Usage:
    python scripts/03_extract_knowledge.py
    python scripts/03_extract_knowledge.py --analyses-dir data/analyses/ --output data/knowledge/knowledge_base.json
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
from rich.table import Table
from rich.tree import Tree

from src.knowledge.extractor import KnowledgeExtractor
from src.knowledge.store import KnowledgeStore

console = Console()

DEFAULT_ANALYSES_DIR = "data/analyses/"
DEFAULT_OUTPUT = "data/knowledge/knowledge_base.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured trading knowledge from video analysis JSON files.",
    )
    parser.add_argument(
        "--analyses-dir",
        type=str,
        default=DEFAULT_ANALYSES_DIR,
        help=f"Directory containing analysis JSON files (default: {DEFAULT_ANALYSES_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output knowledge base JSON path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(argv)


def _load_analyses(analyses_dir: Path) -> list[dict]:
    if not analyses_dir.exists():
        console.print(f"[red]Error:[/red] Analyses directory not found: {analyses_dir}")
        console.print("[dim]Run step 02 first: python scripts/02_analyze_videos.py[/dim]")
        sys.exit(1)

    files = sorted(
        f for f in analyses_dir.glob("*.json")
        if f.name != "processing_state.json"
    )
    if not files:
        console.print(f"[yellow]No analysis files found in {analyses_dir}[/yellow]")
        sys.exit(0)

    analyses: list[dict] = []
    load_errors = 0
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_source_file"] = str(f)
            analyses.append(data)
        except Exception as exc:
            console.print(f"[yellow]Warning: Could not load {f.name}: {exc}[/yellow]")
            load_errors += 1

    if load_errors:
        console.print(f"[dim]{load_errors} file(s) skipped due to load errors.[/dim]")

    return analyses


async def main(args: argparse.Namespace) -> None:
    analyses_dir = Path(args.analyses_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    analyses = _load_analyses(analyses_dir)

    console.print(
        Panel(
            f"[bold cyan]Step 3: Extract Trading Knowledge[/bold cyan]\n"
            f"Analyses: [green]{len(analyses)}[/green] files from [green]{analyses_dir}[/green]\n"
            f"Output:   [green]{output_path}[/green]",
            title="Trading Bot Pipeline",
            border_style="blue",
        )
    )

    extractor = KnowledgeExtractor()
    store = KnowledgeStore(store_path=output_path)

    total_strategies = 0
    extraction_errors = 0

    with console.status("[bold green]Extracting knowledge from analyses..."):
        for i, analysis in enumerate(analyses, 1):
            source_file = analysis.get("_source_file", f"analysis_{i}")
            video_id = analysis.get("video_id", Path(source_file).stem)
            try:
                strategies = extractor.extract_from_analysis(analysis, video_id=video_id)
                total_strategies += len(strategies)
                store.add_strategies(strategies)
                console.print(
                    f"  [{i}/{len(analyses)}] [green]{Path(source_file).name}[/green] "
                    f"-> {len(strategies)} strategy(ies)"
                )
            except Exception as exc:
                extraction_errors += 1
                console.print(
                    f"  [{i}/{len(analyses)}] [red]{Path(source_file).name}[/red] "
                    f"-> Error: {exc}"
                )

    knowledge_base = store.load()

    console.print()
    table = Table(title="Extraction Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Analyses Processed", str(len(analyses)))
    table.add_row("Extraction Errors", f"[red]{extraction_errors}[/red]" if extraction_errors else "0")
    table.add_row("Strategies Found (raw)", str(total_strategies))
    table.add_row("Unique Strategies (merged)", str(len(knowledge_base.strategies)))
    table.add_row("Unique Indicators", str(len(knowledge_base.indicators)))
    table.add_row("General Rules", str(len(knowledge_base.general_rules)))
    table.add_row("Output File", str(output_path))
    console.print(table)

    if knowledge_base.strategies:
        console.print()
        tree = Tree("[bold]Extracted Strategies[/bold]")
        for strategy in knowledge_base.strategies:
            branch = tree.add(
                f"[cyan]{strategy.name}[/cyan] "
                f"(confidence={strategy.confidence_score:.2f}, "
                f"quantifiability={strategy.quantifiability_score:.2f})"
            )
            if strategy.indicators:
                ind_names = [ind.name for ind in strategy.indicators]
                branch.add(f"Indicators: {', '.join(ind_names)}")
            if strategy.timeframes:
                branch.add(f"Timeframes: {', '.join(strategy.timeframes)}")
        console.print(tree)

    console.print(
        f"\n[bold green]Done![/bold green] Extracted knowledge from {len(analyses)} analyses."
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
