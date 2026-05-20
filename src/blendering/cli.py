"""CLI entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console

from .config import load_settings
from .tui.app import BlenderingApp

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


_PROMPT_ARG = typer.Argument(..., help="The scene / task you want Blender to produce.")
_CONFIG_OPT = typer.Option(
    Path("config.yaml"),
    "--config",
    "-c",
    help="Path to config.yaml (defaults to ./config.yaml).",
)


@app.command()
def run(prompt: str = _PROMPT_ARG, config: Path = _CONFIG_OPT) -> None:
    """Run the Actor+Critic agent against Blender via MCP."""
    try:
        settings = load_settings(config)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from exc

    missing = [k for k in os.environ if k.startswith("_BLENDERING_MISSING_")]
    if missing:
        for var in missing:
            envvar = os.environ.get(var, "?")
            console.print(f"[yellow]warn:[/] {var.removeprefix('_BLENDERING_MISSING_')} "
                          f"model is missing API key (env: {envvar})")

    tui = BlenderingApp(settings=settings, user_prompt=prompt)
    tui.run()


def main() -> None:
    # When the user types `blendering "<prompt>"` we don't want them to need a subcommand.
    if len(sys.argv) >= 2 and sys.argv[1] not in {"run", "--help", "-h"}:
        sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    main()
