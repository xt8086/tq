# tq — Project Context

## Overview
**tq** (TurboQuant) is a CLI tool for managing local GGUF model servers with automatic KV cache compression. Written in Python, published as `tq-serve` on PyPI.

- **PyPI**: https://pypi.org/project/tq-serve/ (owner: `wondermotor_ai`)
- **GitHub**: git@github.com:xt8086/tq.git (branch: main)
- **Website**: wondermotor.com (index.html in repo root)
- **Current version**: 0.4.10

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
2. **Code block mode**: For non-tool models. System prompt instructs model to write simple helper calls: `curl()`, `weather()`, `websearch()`, `exec()`. All calls are wrapped in `print(func(arg))` before execution.

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
- 0.4.1 — Fix exec() handler: wrap in print(exec(arg)) instead of stripping quotes; exec() helper now returns stderr + exit code on failure
- 0.4.2 — Same as 0.4.1 (re-published to fix PyPI package)
- 0.4.3 — Fix exec() helper: escape \n as \\n in _AUTO_IMPORTS string (single \n became literal newline, breaking python -c)
- 0.4.4 — Lenient fallback regex for exec() (catches malformed calls with extra junk after closing paren); system prompt: exec() for simple commands only, ```exec``` blocks for pipes/redirects
- 0.4.5 — Shell command fallback: raw shell lines (arp, ifconfig, etc.) auto-wrapped with print(exec(repr(cmd))) when no other pattern matches
- 0.4.6 — 4-step workflow in system prompt: ANALYZE → PLAN → EXECUTE (one attempt) → FINALIZE (stop even on failure). Prevents retry loops.
- 0.4.7 — Tighten workflow: PLAN must be complete (all calls upfront), EXECUTE outputs ALL calls at once, FINALIZE = summary only with zero execution allowed.
- 0.4.8 — Remove shell command fallback (was catching commands in Step 4 explanations and re-triggering execution loops); system prompt: unwrapped commands will NOT execute
- 0.4.9 — Truncate response text at FINALIZE/Step 4 marker before extracting code blocks — prevents code in Step 4 from triggering execution
- 0.4.10 — Enforce Step 3 mandatory (no skipping, no asking permission), Step 2 = describe plan only (no exec calls), never make up results in Step 4

## Design Decisions
- `tq remove` only removes models in tq's model_dir (`~/.tq/models/`), not system-wide models
- Binary (llama-server) is NOT on PyPI — it comes from GitHub Releases (TheTom/llama-cpp-turboquant)
- Models < 3B are classified as "small" → no tool support → code block mode only
- Server auto-stops after idle timeout (default 5 min, configurable)

## Tested Configurations
`tq chat` has been tested with the following models and hardware only:
- **Ministral 3B** (code block mode — non-tool)
- **Qwen3.5 4B** (falls back to code block mode — detected as tool-calling but calls with empty args)

**Hardware**: MacBook Air M1, 8GB RAM

Other models, larger hardware, or tool-calling mode have not been tested. Behavior may vary.
