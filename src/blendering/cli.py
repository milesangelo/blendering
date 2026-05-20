"""CLI entrypoint."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

from .config import load_settings
from .headless import main as run_headless_main
from .tui.app import BlenderingApp
from .utils.logging import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


_PROMPT_ARG = typer.Argument(..., help="The scene / task you want Blender to produce.")
_CONFIG_OPT = typer.Option(
    Path("config.yaml"),
    "--config",
    "-c",
    help="Path to config.yaml (defaults to ./config.yaml).",
)
_HEADLESS_OPT = typer.Option(
    False,
    "--headless",
    help="Stream events to stdout instead of launching the TUI (for scripts / CI).",
)


@app.command()
def run(
    prompt: str = _PROMPT_ARG,
    config: Path = _CONFIG_OPT,
    headless: bool = _HEADLESS_OPT,
) -> None:
    """Run the Actor+Critic agent against Blender via MCP."""
    log_path = setup_logging()
    console.print(f"[dim]log: {log_path}[/]")
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

    if headless:
        raise typer.Exit(code=run_headless_main(settings, prompt))

    tui = BlenderingApp(settings=settings, user_prompt=prompt)
    tui.run()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
