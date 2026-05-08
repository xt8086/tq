from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from . import config as cfg
from .scanner import scan_models, find_model, resolve_model_path
from .parser import build_model_metadata, GGUFParserError
from .hardware import detect_hardware
from .recommender import recommend
from .server import start_server, stop_server, get_server_status, load_state, _find_binary
from .hf import search_models as hf_search, download_model
from .installer import install_binary, get_platform_tag
from .types import ServerConfig, CacheType

console = Console()


def _short_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home):]
    return path


def cmd_list(args):
    model_dir = cfg.get_model_dir()
    models = scan_models(model_dir, system_wide=True)

    if not models:
        console.print(f"[dim]No GGUF models found in {model_dir}[/dim]")
        console.print("[dim]Use 'tq download <model>' to fetch one, or set model_dir in config.[/dim]")
        return

    for i, m in enumerate(models, 1):
        quant = m.quant_type.value if m.quant_type.value != "UNKNOWN" else ""
        mm = " [bold green]multimodal[/]" if m.is_multimodal else ""
        line = f"{i}. {m.display_name}  ({m.size_gb:.1f}G"
        if quant:
            line += f", {quant}"
        line += f"){mm}"
        console.print(line)
        console.print(f"   {m.path}")
        if i < len(models):
            console.print()


def cmd_search(args):
    results = hf_search(args.query, limit=args.limit)

    if not results:
        console.print(f"[dim]No results for '{args.query}'[/dim]")
        return

    table = Table(title=f"Search: {args.query}", box=box.ROUNDED)
    table.add_column("Model ID", style="cyan")
    table.add_column("Downloads", justify="right", style="green")
    table.add_column("GGUF Files", justify="right", style="yellow")

    for r in results:
        table.add_row(
            r["id"],
            f"{r.get('downloads', 0):,}",
            str(r["total_gguf_files"]),
        )

    console.print(table)
    console.print("\n[dim]Use 'tq download <model_id>' to download.[/dim]")


def cmd_download(args):
    model_dir = cfg.get_model_dir()
    console.print(f"[bold]Downloading[/bold] {args.model}")

    try:
        path = download_model(
            model_id=args.model,
            filename=args.file,
            model_dir=model_dir,
            verify_hash=not args.skip_verify,
        )
        console.print(f"[green]Downloaded:[/green] {path}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def cmd_serve(args):
    model_dir = cfg.get_model_dir()
    port = args.port or cfg.get_port()
    host = args.host or cfg.get_host()
    binary_path = cfg.get_binary_path()

    model_arg = args.model
    if not model_arg:
        models = scan_models(model_dir, system_wide=True)
        if not models:
            console.print("[dim]No GGUF models found.[/dim]")
            console.print("[dim]Use 'tq download <model>' to fetch one.[/dim]")
            sys.exit(1)
        if len(models) == 1:
            model_arg = "1"
        else:
            for i, m in enumerate(models, 1):
                mm = " [bold green]multimodal[/]" if m.is_multimodal else ""
                console.print(f"  {i}. {m.display_name}  ({m.size_gb:.1f}G){mm}")
            try:
                model_arg = input("  Serve which model? [number]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                sys.exit(0)

    model_path = resolve_model_path(model_dir, model_arg)
    if not model_path:
        console.print(f"[red]Model not found:[/red] {args.model}")
        console.print(f"[dim]Searched in: {model_dir}[/dim]")
        sys.exit(1)

    try:
        meta = build_model_metadata(model_path)
    except (GGUFParserError, OSError) as e:
        console.print(f"[yellow]Warning:[/yellow] Could not parse GGUF metadata: {e}")
        from .types import ModelMetadata, QuantType
        meta = ModelMetadata(
            name=os.path.basename(model_path),
            path=model_path,
            size_bytes=os.path.getsize(model_path),
        )

    from .scanner import _find_mmproj
    mmproj = _find_mmproj(model_path)
    if mmproj:
        meta.mmproj_path = mmproj
        meta.is_multimodal = True

    hw = detect_hardware()
    ctx = args.context

    rec = recommend(meta, hw, context_length=ctx)

    console.print(Panel(
        f"[bold]Model:[/bold]  {meta.display_name}\n"
        f"[bold]Size:[/bold]    {meta.size_gb:.1f} GB\n"
        f"[bold]Quant:[/bold]   {meta.quant_type.value}\n"
        f"[bold]Hardware:[/bold] {hw.gpu_name} ({hw.ram_gb:.0f} GB)\n"
        + (f"[bold]Multimodal:[/bold] Yes (mmproj detected)\n" if meta.is_multimodal else "")
        + f"\n[bold]TQ Config:[/bold]\n"
        f"  ctk = {rec.cache_type_k.value}\n"
        f"  ctv = {rec.cache_type_v.value}\n"
        f"  boundary_v = {rec.boundary_v}\n"
        f"  sparse_v = {rec.sparse_v}\n"
        f"  context = {rec.context_length}\n"
        f"\n[bold]Reasoning:[/bold]\n" +
        "\n".join(f"  - {r}" for r in rec.reasoning),
        title="TurboQuant Recommendation",
        box=box.ROUNDED,
    ))

    extra_flags = []
    if args.override_flags:
        extra_flags = args.override_flags.split()

    api_key = None
    idle_timeout = cfg.load_config().get("idle_timeout", 300)

    server_config = ServerConfig(
        model_path=model_path,
        port=port,
        host=host,
        tq=rec,
        extra_flags=extra_flags,
        api_key=api_key,
        idle_timeout=idle_timeout,
        mmproj_path=meta.mmproj_path if meta.is_multimodal else None,
    )

    try:
        binary = _find_binary(binary_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    cmd_str = " ".join(server_config.to_command(binary))

    if args.dry_run:
        console.print(f"\n[bold]Command:[/bold]\n{cmd_str}")
        return

    existing = load_state()
    if existing:
        console.print("[yellow]A server is already running.[/yellow] Use 'tq stop' first.")
        sys.exit(1)

    if host != "127.0.0.1":
        console.print(f"[bold red]WARNING:[/bold red] Server will bind to {host}, exposing it on the network.")
        console.print("[dim]Consider using --host 127.0.0.1 for local-only access.[/dim]")

    console.print(f"[bold]Starting server on {host}:{port}...[/bold]")
    console.print(f"[dim]Command: {cmd_str}[/dim]")

    state = start_server(server_config)

    if state:
        console.print(f"[green]Server started[/green] (PID: {state.pid})")
        console.print(f"[bold]API:[/bold] http://{host}:{port}/v1/chat/completions")
        console.print(f"[dim]OpenAI-compatible, no auth needed. Usage:\n   curl http://{host}:{port}/v1/chat/completions -d '{{\"model\":\"...\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'\n   Or set base_url='http://{host}:{port}/v1' in the OpenAI Python SDK.[/dim]")
        console.print(f"[dim]Logs: ~/.tq/logs/[/dim]")


def cmd_status(args):
    info = get_server_status()
    if not info:
        console.print("[dim]No server running.[/dim]")
        return

    status_color = "green" if info["healthy"] else "red"
    status_text = "HEALTHY" if info["healthy"] else "UNHEALTHY"

    idle_mins = info.get("idle_timeout", 300) // 60
    idle_secs = info.get("idle_timeout", 300)

    console.print(Panel(
        f"[bold]Model:[/bold]    {info['model']}\n"
        f"[bold]PID:[/bold]       {info['pid']}\n"
        f"[bold]Address:[/bold]  http://{info['host']}:{info['port']}\n"
        f"[bold]Status:[/bold]  [{status_color}]{status_text}[/{status_color}]\n"
        f"[bold]Uptime:[/bold]   {info['uptime_seconds']}s\n"
        f"[bold]Idle timeout:[/bold] {idle_mins}m ({idle_secs}s)",
        title="Server Status",
        box=box.ROUNDED,
    ))


def cmd_stop(args):
    state = load_state()
    if not state:
        console.print("[dim]No server running.[/dim]")
        return

    console.print(f"[bold]Stopping server[/bold] (PID: {state.pid})...")
    if stop_server():
        console.print("[green]Server stopped.[/green]")
    else:
        console.print("[yellow]Server process not found (may have already stopped).[/yellow]")


def cmd_logs(args):
    log_dir = os.path.expanduser("~/.tq/logs")
    if not os.path.isdir(log_dir):
        console.print("[dim]No logs found.[/dim]")
        return

    logs = sorted(
        [f for f in os.listdir(log_dir) if f.endswith(".log")],
        reverse=True,
    )

    if not logs:
        console.print("[dim]No logs found.[/dim]")
        return

    latest = os.path.join(log_dir, logs[0])
    lines = args.lines or 50

    try:
        with open(latest, errors="replace") as f:
            all_lines = f.readlines()
        for line in all_lines[-lines:]:
            print(line.rstrip())
    except Exception as e:
        print(f"Error reading log: {e}")


def cmd_validate(args):
    model_dir = cfg.get_model_dir()
    model_path = resolve_model_path(model_dir, args.model)
    if not model_path:
        console.print(f"[red]Model not found:[/red] {args.model}")
        sys.exit(1)

    try:
        meta = build_model_metadata(model_path)
    except GGUFParserError as e:
        console.print(f"[red]Invalid GGUF file:[/red] {e}")
        sys.exit(1)

    hw = detect_hardware()
    rec = recommend(meta, hw)

    budget_gb = (hw.ram_gb * 0.75) if hw.is_apple_silicon else (hw.vram_gb * 0.85)
    fits = meta.size_gb < budget_gb

    console.print(Panel(
        f"[bold]Model:[/bold]  {meta.display_name}\n"
        f"[bold]Size:[/bold]    {meta.size_gb:.1f} GB\n"
        f"[bold]Quant:[/bold]   {meta.quant_type.value}\n"
        f"[bold]Arch:[/bold]    {meta.architecture or 'unknown'}\n"
        f"[bold]Layers:[/bold]  {meta.num_layers or '?'}\n"
        f"[bold]Context:[/bold] {meta.context_length or '?'}\n"
        f"\n[bold]Hardware:[/bold] {hw.gpu_name}\n"
        f"[bold]RAM:[/bold]     {hw.ram_gb:.0f} GB\n"
        f"[bold]VRAM:[/bold]    {hw.vram_gb:.0f} GB\n"
        f"\n[bold]Budget:[/bold]  {budget_gb:.1f} GB available → {'✓ FITS' if fits else '✗ TOO LARGE'}\n"
        f"\n[bold]Recommended TQ:[/bold]\n"
        + "\n".join(f"  - {r}" for r in rec.reasoning) +
        f"\n  ctk={rec.cache_type_k.value} ctv={rec.cache_type_v.value}"
        f" ctx={rec.context_length}",
        title="Validation",
        box=box.ROUNDED,
    ))

    if not fits:
        console.print("[red]Model may not fit in available memory.[/red]")


def cmd_install(args):
    console.print(f"[bold]Installing TurboQuant+ llama-server[/bold]")
    console.print(f"[dim]Platform: {platform.system()} {platform.machine()}[/dim]")

    try:
        path = install_binary(force=args.force)
        cfg.set_value("binary_path", path)
        console.print(f"[green]Done![/green] Binary set to: {path}")
        console.print("Run [bold]tq doctor[/bold] to verify.")
    except Exception as e:
        console.print(f"[red]Install failed:[/red] {e}")
        sys.exit(1)


def cmd_doctor(args):
    binary_path = cfg.get_binary_path()
    issues = []

    try:
        binary = _find_binary(binary_path)
        console.print(f"[green]✓[/green] llama-server found: {binary}")
    except FileNotFoundError:
        console.print("[red]✗[/red] llama-server binary not found")
        issues.append("binary")

    model_dir = cfg.get_model_dir()
    if os.path.isdir(model_dir):
        count = sum(1 for _, _, files in os.walk(model_dir) for f in files if f.endswith(".gguf"))
        console.print(f"[green]✓[/green] Model dir: {model_dir} ({count} models)")
    else:
        console.print(f"[yellow]⚠[/yellow] Model dir does not exist: {model_dir}")
        issues.append("model_dir")

    hw = detect_hardware()
    console.print(f"[green]✓[/green] GPU: {hw.gpu_name}")
    console.print(f"[green]✓[/green] RAM: {hw.ram_gb:.0f} GB")

    config_path = cfg.CONFIG_FILE
    if os.path.isfile(config_path):
        import stat
        mode = os.stat(config_path).st_mode & 0o777
        if mode & 0o077:
            console.print(f"[red]✗[/red] Config file permissions too open: {oct(mode)} (should be 0600)")
            issues.append("config_perms")
        else:
            console.print(f"[green]✓[/green] Config permissions OK ({oct(mode)})")
    else:
        console.print("[yellow]⚠[/yellow] No config file yet (will be created on first use)")

    if issues:
        console.print(f"\n[yellow]Issues found: {len(issues)}. Fix before using tq serve.[/yellow]")
    else:
        console.print("\n[green]All checks passed.[/green]")


def cmd_chat(args):
    try:
        from .chat.repl import ChatSession
        from .chat.permissions import PermissionConfig
    except ImportError:
        console.print("[red]Chat dependencies not installed.[/red]")
        console.print("[dim]Install with: pip install tq-serve[chat][/dim]")
        sys.exit(1)

    perms = PermissionConfig.defaults()
    if args.yolo:
        from .chat.permissions import PermissionAction
        perms = PermissionConfig.from_dict({"*": PermissionAction.ALLOW})
    if args.ask_all:
        from .chat.permissions import PermissionAction
        perms = PermissionConfig.from_dict({"*": PermissionAction.ASK})

    base_url = f"http://{args.host}:{args.port}"

    session = ChatSession(
        base_url=base_url,
        model=args.model or "",
        workdir=args.workdir or os.getcwd(),
        no_serve=args.no_serve,
        system_prompt=args.system or "",
        perms=perms,
    )
    session.start()


def cmd_config(args):
    if args.action == "show" or args.action is None:
        config = cfg.load_config()
        display = cfg.show_config(config)
        table = Table(title="Configuration", box=box.ROUNDED)
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        for k, v in display.items():
            table.add_row(k, v)
        console.print(table)
    elif args.action == "get":
        config = cfg.load_config()
        val = config.get(args.key, "")
        if "token" in args.key.lower() or "key" in args.key.lower():
            from .security import redact_token
            val = redact_token(str(val))
        console.print(f"{args.key} = {val}")
    elif args.action == "set":
        cfg.set_value(args.key, args.value)
        console.print(f"[green]Set {args.key}[/green]")
    elif args.action == "path":
        console.print(cfg.CONFIG_FILE)
    elif args.action == "reset":
        cfg.save_config(dict(cfg._DEFAULTS))
        console.print("[green]Config reset to defaults.[/green]")


def main():
    parser = argparse.ArgumentParser(
        prog="tq",
        description="TurboQuant model server manager",
    )
    sub = parser.add_subparsers(dest="command")

    ls = sub.add_parser("list", help="List local GGUF models")
    ls.add_argument("-d", "--model-dir", help="Override model directory")
    ls.add_argument("-s", "--system", action="store_true", help="Search system-wide for GGUF models")

    sr = sub.add_parser("search", help="Search HuggingFace for GGUF models")
    sr.add_argument("query", help="Search query")
    sr.add_argument("--limit", type=int, default=20, help="Max results")

    dl = sub.add_parser("download", help="Download a GGUF model from HuggingFace")
    dl.add_argument("model", help="Model ID (e.g. TheBloke/Llama-2-7B-GGUF)")
    dl.add_argument("--file", help="Specific GGUF filename to download")
    dl.add_argument("--skip-verify", action="store_true", help="Skip SHA256 verification")

    sv = sub.add_parser("serve", help="Launch llama-server with optimal TQ settings")
    sv.add_argument("model", nargs="?", default="", help="Model name, path, or list number")
    sv.add_argument("-p", "--port", type=int, help="Port (default: 8080)")
    sv.add_argument("--host", default=None, help="Bind address (default: 127.0.0.1)")
    sv.add_argument("-c", "--context", type=int, help="Context length override")
    sv.add_argument("--dry-run", action="store_true", help="Print command without running")
    sv.add_argument("--override-flags", help="Extra llama-server flags (quoted string)")
    sv.add_argument("-d", "--model-dir", help="Override model directory")

    st = sub.add_parser("status", help="Check if server is running")

    sp = sub.add_parser("stop", help="Stop the running server")

    lg = sub.add_parser("logs", help="View server logs")
    lg.add_argument("-n", "--lines", type=int, default=50, help="Number of lines")

    val = sub.add_parser("validate", help="Validate a model and show TQ recommendation")
    val.add_argument("model", help="Model name or path")
    val.add_argument("-d", "--model-dir", help="Override model directory")

    inst = sub.add_parser("install", help="Download and install TurboQuant+ llama-server binary")
    inst.add_argument("--force", action="store_true", help="Reinstall even if already installed")

    doc = sub.add_parser("doctor", help="Verify setup and configuration")

    cf = sub.add_parser("config", help="Show/edit configuration")
    cf.add_argument("action", choices=["show", "get", "set", "reset", "path"], nargs="?", default="show")
    cf.add_argument("key", nargs="?", help="Config key")
    cf.add_argument("value", nargs="?", help="Config value")

    ch = sub.add_parser("chat", help="Interactive coding agent (local AI)")
    ch.add_argument("-m", "--model", help="Model name to use")
    ch.add_argument("-p", "--port", type=int, default=8080, help="Server port")
    ch.add_argument("--host", default="127.0.0.1", help="Server host")
    ch.add_argument("-w", "--workdir", help="Working directory (default: cwd)")
    ch.add_argument("--no-serve", action="store_true", help="Don't auto-start server")
    ch.add_argument("-s", "--system", help="Custom system prompt")
    ch.add_argument("--yolo", action="store_true", help="Allow all tool actions without asking")
    ch.add_argument("--ask-all", action="store_true", help="Ask permission for all actions")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    cfg.init_config()

    if args.command in ("list", "serve", "validate") and hasattr(args, "model_dir") and args.model_dir:
        cfg.set_value("model_dir", args.model_dir)

    commands = {
        "list": cmd_list,
        "search": cmd_search,
        "download": cmd_download,
        "serve": cmd_serve,
        "status": cmd_status,
        "stop": cmd_stop,
        "logs": cmd_logs,
        "validate": cmd_validate,
        "install": cmd_install,
        "doctor": cmd_doctor,
        "config": cmd_config,
        "chat": cmd_chat,
    }

    fn = commands.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()