# tq — Project Context

## Overview
**tq** (TurboQuant) is a CLI tool for managing local GGUF model servers with automatic KV cache compression. Written in Python, published as `tq-serve` on PyPI.

- **PyPI**: https://pypi.org/project/tq-serve/ (owner: `wondermotor_ai`)
- **GitHub**: git@github.com:xt8086/tq.git (branch: main)
- **Website**: wondermotor.com (index.html in repo root)
- **Current version**: 0.4.0

## Architecture

### Key Files
| File | Purpose |
|------|---------|
| `tq/cli.py` | All CLI commands (argparse subparsers + `cmd_xxx` handlers) |
| `tq/parser.py` | GGUF metadata parser, quant detection, tool support detection (`_SMALL_MODEL_PATTERNS`, `_TOOL_CAPABLE_ARCHES`) |
| `tq/scanner.py` | Model discovery (`scan_models`, `resolve_model_path`, `find_model`, `_find_mmproj`) |
| `tq/server.py` | Server lifecycle (`start_server`, `stop_server`, `load_state`, `_find_binary`) |
| `tq/config.py` | Config CRUD (TOML at `~/.tq/config.toml`) |
| `tq/security.py` | Path validation (`validate_gguf_path`), secure file writes, SHA256 |
| `tq/hf.py` | HuggingFace search/download, hash verification, `remove_hash_for_file` |
| `tq/installer.py` | Download TurboQuant+ llama-server binary from GitHub Releases |
| `tq/recommender.py` | Auto-config: quant → cache type, context length budget |
| `tq/hardware.py` | GPU/RAM detection (Apple Silicon, NVIDIA, AMD) |
| `tq/types.py` | Dataclasses: `ModelMetadata`, `ServerState`, `TQRecommendation`, etc. |
| `tq/chat/repl.py` | Interactive chat REPL, code block execution, system prompts |
| `tq/chat/client.py` | OpenAI-compatible streaming chat client (httpx) |
| `tq/chat/tools.py` | Tool registry (bash, read, edit, write, glob, grep, websearch, etc.) |
| `tq/chat/permissions.py` | Permission system (allow/ask/deny per tool) |
| `tq/chat/render.py` | Rich console rendering helpers |

### How Commands Work
Each command = `cmd_xxx(args)` function + argparse subparser in `main()` + entry in `commands` dict.

### tq chat Modes (two modes, auto-detected)
1. **Tool-calling mode**: For models with OpenAI tool support (detected by `_detect_tool_support` in parser.py). Uses full tool set (bash, read, write, etc.)
2. **Code block mode**: For non-tool models. System prompt instructs model to write `` ```exec `` blocks. Python code auto-executed with built-in helpers: `curl()`, `weather()`, `websearch()`

Tool support detection logic (`_detect_tool_support`):
- Small models (by name pattern in `_SMALL_MODEL_PATTERNS`) → NONE
- Chat template has `tool_calls` + `function` → OPENAI
- Architecture in `_TOOL_CAPABLE_ARCHES` → OPENAI
- Otherwise → NONE

### Key Paths
- Config: `~/.tq/config.toml`
- Models: `~/.tq/models/`
- Binary: `~/.tq/bin/`
- Logs: `~/.tq/logs/`
- Server state: `~/.tq/server.json`
- Hash cache: `~/.tq/hashes.json`
- Chat history: `~/.tq/chat_history`

### Security Model
- `validate_gguf_path(path, model_dir)` — ensures path is within model_dir (prevents traversal)
- `tq remove` only allows deleting files inside `~/.tq/models/` (refuses lmstudio/ollama models)
- `tq remove` refuses if model is currently being served (checks `load_state()`)
- Config and state files stored with 0600 permissions

## Release Process
1. Make changes, commit to main
2. Push: `git push origin main`
3. Bump version in `pyproject.toml`
4. Build: `rm -f dist/* && ~/tq/.venv/bin/python -m build`
5. Publish: `~/tq/.venv/bin/twine upload dist/* -u __token__ -p 'pypi-token'`
   - Token from PyPI account: `wondermotor_ai`

## Version History
- 0.1.0 — Initial release
- 0.2.0 — Chat, tools, websearch, weather helpers
- 0.3.0 — Uninstall command, fix install.sh symlink
- 0.4.0 — `tq remove` command, fix GitHub URLs to xt8086/tq, update small model patterns (Qwen3.5)

## Design Decisions
- `tq remove` only removes models in tq's model_dir (`~/.tq/models/`), not system-wide models
- Binary (llama-server) is NOT on PyPI — it comes from GitHub Releases (TheTom/llama-cpp-turboquant)
- Models < 3B are classified as "small" → no tool support → code block mode only
- Server auto-stops after idle timeout (default 5 min, configurable)
