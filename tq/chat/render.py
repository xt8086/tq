from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text


console = Console()


def render_markdown(text: str):
    console.print(Markdown(text))


def render_streaming(token_generator, stop_event=None):
    buffer = ""
    try:
        with Live(console=console, refresh_per_second=8, vertical_overflow="visible") as live:
            for token in token_generator:
                if stop_event and stop_event.is_set():
                    break
                buffer += token
                live.update(Markdown(buffer))
    except KeyboardInterrupt:
        pass
    return buffer


def render_tool_call(name: str, args: dict):
    arg_str = " ".join(f"{k}={v}" for k, v in args.items())
    if len(arg_str) > 120:
        arg_str = arg_str[:117] + "..."
    console.print(f"  [dim]→ {name}({arg_str})[/dim]")


def render_tool_result(name: str, result: str, error: bool = False):
    color = "red" if error else "dim"
    prefix = "✗" if error else "←"
    preview = result[:200].replace("", "")
    if len(result) > 200:
        preview += "..."
    console.print(f"  [{color}]{prefix} {name}: {preview}[/{color}]")


def render_status(model: str, base_url: str, context_info: str = ""):
    console.print(f"  [green]●[/green] Connected to [bold]{base_url}[/bold]")
    console.print(f"  [green]●[/green] Model: [bold]{model}[/bold]")
    if context_info:
        console.print(f"  [green]●[/green] {context_info}")


def render_error(msg: str):
    console.print(f"  [red]✗[/red] {msg}")


def render_info(msg: str):
    console.print(f"  [dim]ℹ {msg}[/dim]")


def render_divider():
    console.print("[dim]  ─────────────────────────────────────────────────────[/dim]")
