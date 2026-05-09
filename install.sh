#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${GREEN}[tq]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[tq]${RESET} $*"; }
error() { echo -e "${RED}[tq]${RESET} $*" >&2; exit 1; }

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" != "Darwin" ] && [ "$OS" != "Linux" ]; then
    error "Unsupported OS: $OS. Only macOS and Linux are supported."
fi

if [ "$ARCH" != "arm64" ] && [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
    error "Unsupported architecture: $ARCH"
fi

if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python 3.10+ first."
fi

PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required. Found: ${PY_MAJOR}.${PY_MINOR}"
fi

info "Installing tq — TurboQuant model server manager"
info "Platform: ${OS} ${ARCH}"

TQ_VENV="$HOME/.tq/venv"
TQ_BIN="$TQ_VENV/bin/tq"
TQ_LINK_DIR="$HOME/.local/bin"
TQ_LINK="$TQ_LINK_DIR/tq"

info "Creating virtual environment..."
python3 -m venv "$TQ_VENV"

info "Installing tq package..."
"$TQ_VENV/bin/pip" install -q "tq-serve[chat]"

info "Installing TurboQuant+ llama-server binary..."
"$TQ_BIN" install

mkdir -p "$TQ_LINK_DIR"
ln -sf "$TQ_BIN" "$TQ_LINK"

NEED_PATH_UPDATE=false
case ":$PATH:" in
    *":$TQ_LINK_DIR:"*) ;;
    *) NEED_PATH_UPDATE=true ;;
esac

if [ "$NEED_PATH_UPDATE" = true ]; then
    SHELL_RC="$HOME/.zshrc"
    if [ "$SHELL" = "bash" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ "$SHELL" = "fish" ]; then
        SHELL_RC="$HOME/.config/fish/config.fish"
    fi

    if [ "$SHELL" = "fish" ]; then
        if ! grep -q 'tq-serve' "$SHELL_RC" 2>/dev/null; then
            echo '' >> "$SHELL_RC"
            echo '# tq — TurboQuant model server manager' >> "$SHELL_RC"
            echo 'set -gx PATH $PATH '"'$TQ_LINK_DIR'" >> "$SHELL_RC"
        fi
    else
        if ! grep -q 'tq-serve' "$SHELL_RC" 2>/dev/null; then
            echo '' >> "$SHELL_RC"
            echo '# tq — TurboQuant model server manager' >> "$SHELL_RC"
            echo 'export PATH="$PATH:'"$TQ_LINK_DIR"'"' >> "$SHELL_RC"
        fi
    fi
fi

export PATH="$PATH:$TQ_LINK_DIR"

mkdir -p "$HOME/.tq/models"

echo ""
info "${BOLD}Installation complete!${RESET}"
echo ""
echo "  tq doctor          # Verify setup"
echo "    tq list            # List local GGUF models"
echo "    tq search <query>  # Search HuggingFace"
echo "    tq download <id>   # Download a model"
echo "    tq serve 1         # Launch with auto-configured TurboQuant"
echo "    tq chat            # Interactive coding agent (local AI)"
if [ "$NEED_PATH_UPDATE" = true ]; then
    echo ""
    warn "Open a new terminal for tq to be in PATH, or run:"
    echo "  source $SHELL_RC"
fi
