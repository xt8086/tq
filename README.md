# tq — TurboQuant Model Server Manager

Auto-configured llama-server with KV cache compression.

## Install

```bash
# One-liner (macOS / Linux)
curl -fsSL https://raw.githubusercontent.com/xt8086/tq/main/install.sh | bash

# Or via pip (then run 'tq install' for the binary)
pip install tq-serve
tq install
```

> **Note:** `tq-serve` is a pure Python package — it does NOT include the `llama-server` binary.
> The binary comes from [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant),
> a fork of llama.cpp with TurboQuant KV cache compression built in.
> `tq install` downloads the correct binary for your platform (macOS Metal, Linux CUDA/ROCm).

## Quick Start

```bash
tq doctor                        # Verify setup
tq list                          # List local GGUF models
tq search "qwen2.5 coder 7b"    # Search HuggingFace (numbered, pick # to download)
tq serve 1                       # Launch with auto-configured TurboQuant
tq chat                          # Interactive coding agent
```

## How It Works

`tq serve` automatically:

1. Detects your hardware (GPU, RAM)
2. Parses model metadata (quant type, layers, context length)
3. Calculates optimal TurboQuant cache settings
4. Launches llama-server with the right flags

Example: A Q4_K_M model on Apple M1 with 8GB RAM gets:
- `ctk=q8_0` (protect K cache)
- `ctv=turbo4` (compress V cache 3.8x)
- Context capped to safe memory limit
- Idle auto-stop after 5 min

## Commands

| Command | Description |
|---------|-------------|
| `tq list` | List local GGUF models |
| `tq search <query>` | Search HuggingFace for GGUF models (numbered, with download prompt) |
| `tq download <model>` | Download a model from HuggingFace |
| `tq remove <model>` | Remove a downloaded model |
| `tq serve <model>` | Launch with auto TurboQuant config |
| `tq serve 1` | Serve by list number |
| `tq serve 1 --dry-run` | Show command without running |
| `tq status` | Check if server is running |
| `tq stop` | Stop the server |
| `tq logs` | View server logs |
| `tq validate <model>` | Pre-flight check |
| `tq install` | Download TurboQuant+ binary (from TheTom/llama-cpp-turboquant) |
| `tq doctor` | Verify setup |
| `tq config show` | Show/edit configuration |
| `tq chat` | Interactive coding agent (local AI) |

## API

The server exposes an OpenAI-compatible API:

```
POST http://127.0.0.1:8080/v1/chat/completions
```

No auth needed. Works with any OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="your-model.gguf",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Configuration

Config stored at `~/.tq/config.toml`:

```bash
tq config show              # Show all settings
tq config set port 9090     # Change port
tq config set idle_timeout 600  # 10 min idle timeout (0 to disable)
```

## TurboQuant Cache Types

| Type | Bits | Compression | Use Case |
|------|------|-------------|----------|
| f16 | 16 | 1x | No compression (baseline) |
| q8_0 | 8 | 2x | Safe for K cache |
| turbo4 | 4.25 | 3.8x | Best quality/compression for V |
| turbo3 | 3.25 | 4.9x | Aggressive, for large models |

## Requirements

- Python 3.10+
- macOS (Apple Silicon) or Linux (x86_64 with NVIDIA/AMD GPU)
- ~2GB free RAM minimum (depends on model)
