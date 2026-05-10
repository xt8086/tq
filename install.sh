#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${GREEN}✓${RESET} $1"; }
warn()  { echo -e "${YELLOW}⚠${RESET} $1"; }
error() { echo -e "${RED}✗${RESET} $1"; }
step()  { echo -e "${BOLD}==>${RESET} $1"; }

step "Installing tq — TurboQuant Model Server Manager"

# --- pip install ---
step "Installing tq-serve via pip..."
if command -v pip3 &>/dev/null; then
    PIP=pip3
elif command -v pip &>/dev/null; then
    PIP=pip
else
    error "pip not found. Install Python 3.10+ first: https://python.org"
    exit 1
fi

if "$PIP" install --upgrade tq-serve 2>&1; then
    info "tq-serve installed"
else
    # Try with --break-system-packages for externally managed environments
    warn "pip install failed — retrying with --break-system-packages..."
    if "$PIP" install --upgrade --break-system-packages tq-serve 2>&1; then
        info "tq-serve installed"
    else
        error "pip install failed. Try: python3 -m pip install --user tq-serve"
        exit 1
    fi
fi

# --- verify tq command ---
if ! command -v tq &>/dev/null; then
    warn "tq not on PATH. You may need to add your Python bin directory to PATH."
    echo -e "${DIM}  Usually: export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}"
fi

# --- install binary ---
step "Downloading TurboQuant+ llama-server binary..."
if command -v tq &>/dev/null; then
    if tq install 2>&1; then
        info "llama-server binary installed"
    else
        warn "Binary install failed or not available for your platform."
        echo -e "${DIM}  You can run 'tq install' manually later.${RESET}"
    fi
else
    warn "tq command not found on PATH — skipping binary install."
    echo -e "${DIM}  Run 'tq install' after adding tq to your PATH.${RESET}"
fi

# --- done ---
echo ""
step "Running tq doctor..."
if command -v tq &>/dev/null; then
    tq doctor || true
else
    warn "Cannot run 'tq doctor' — tq not on PATH."
fi

echo ""
echo -e "${BOLD}Done!${RESET} Quick start:"
echo -e "  ${DIM}tq search \"qwen 3b\"${RESET}       # Find a model"
echo -e "  ${DIM}tq serve 1${RESET}                   # Launch server"
echo -e "  ${DIM}tq chat${RESET}                      # Start chatting"
