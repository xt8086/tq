from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from typing import Optional

import psutil

from .types import ServerConfig, ServerState
from .security import ensure_secure_dir, secure_write_json

STATE_FILE = os.path.join(os.path.expanduser("~/.tq"), "server.json")
WATCHER_PID_FILE = os.path.join(os.path.expanduser("~/.tq"), "watcher.pid")


def start_server(config: ServerConfig, dry_run: bool = False) -> Optional[ServerState]:
    binary = _find_binary("")

    if dry_run:
        return None

    cmd = config.to_command(binary)

    env = os.environ.copy()
    if config.api_key:
        env["LLAMA_API_KEY"] = config.api_key

    log_dir = os.path.expanduser("~/.tq/logs")
    ensure_secure_dir(log_dir)
    log_path = os.path.join(log_dir, f"server-{int(time.time())}.log")
    log_fh = open(log_path, "w")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )

    session_token = secrets.token_hex(16)

    state = ServerState(
        pid=proc.pid,
        port=config.port,
        host=config.host,
        model_path=config.model_path,
        binary_path=binary,
        session_token=session_token,
        started_at=time.time(),
        idle_timeout=config.idle_timeout,
    )

    _save_state(state)

    if not _wait_for_health(config.host, config.port, timeout=30):
        stop_server()
        raise RuntimeError(f"Server failed to start within 30s. Check log: {log_path}")

    _start_watcher(state)

    return state


def stop_server() -> bool:
    _stop_watcher()

    state = load_state()
    if not state:
        return False

    if not _validate_pid(state.pid, state.binary_path):
        _clear_state()
        return False

    try:
        proc = psutil.Process(state.pid)
        children = proc.children(recursive=True)
        proc.terminate()

        gone, alive = psutil.wait_procs([proc] + children, timeout=10)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass

    except psutil.NoSuchProcess:
        pass
    finally:
        _clear_state()

    return True


def get_server_status() -> Optional[dict]:
    state = load_state()
    if not state:
        return None

    if not _validate_pid(state.pid, state.binary_path):
        _clear_state()
        return None

    healthy = _check_health(state.host, state.port)
    uptime = time.time() - state.started_at

    return {
        "pid": state.pid,
        "port": state.port,
        "host": state.host,
        "model": os.path.basename(state.model_path),
        "healthy": healthy,
        "uptime_seconds": round(uptime),
        "idle_timeout": state.idle_timeout,
    }


def load_state() -> Optional[ServerState]:
    if not os.path.isfile(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        return ServerState(
            pid=data["pid"],
            port=data["port"],
            host=data["host"],
            model_path=data["model_path"],
            binary_path=data["binary_path"],
            session_token=data["session_token"],
            started_at=data["started_at"],
            idle_timeout=data.get("idle_timeout", 300),
        )
    except (json.JSONDecodeError, KeyError):
        return None


def _save_state(state: ServerState) -> None:
    ensure_secure_dir(os.path.dirname(STATE_FILE))
    data = json.dumps({
        "pid": state.pid,
        "port": state.port,
        "host": state.host,
        "model_path": state.model_path,
        "binary_path": state.binary_path,
        "session_token": state.session_token,
        "started_at": state.started_at,
        "idle_timeout": state.idle_timeout,
    })
    secure_write_json(STATE_FILE, data)


def _clear_state() -> None:
    if os.path.isfile(STATE_FILE):
        os.unlink(STATE_FILE)


def _validate_pid(pid: int, expected_binary: str) -> bool:
    try:
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline()).lower()
        return "llama-server" in cmdline or os.path.basename(expected_binary).lower() in cmdline
    except psutil.NoSuchProcess:
        return False


def _find_binary(hint: str = "") -> str:
    import shutil

    if hint and os.path.isfile(hint):
        return hint

    tq_bin = os.path.expanduser("~/.tq/bin")
    if os.path.isdir(tq_bin):
        for root, _dirs, files in os.walk(tq_bin):
            for f in files:
                if f == "llama-server":
                    return os.path.join(root, f)

    name = shutil.which("llama-server")
    if name:
        return name

    name = shutil.which("llama-server-turboquant")
    if name:
        return name

    common_paths = [
        os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
        "/usr/local/bin/llama-server",
        "/opt/homebrew/bin/llama-server",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "llama-server binary not found. Install turboquant_plus or set binary_path in config."
    )


def _check_health(host: str, port: int) -> bool:
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/health"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _wait_for_health(host: str, port: int, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _check_health(host, port):
            return True
        time.sleep(0.5)
    return False


def _start_watcher(state: ServerState) -> None:
    if state.idle_timeout <= 0:
        return

    watcher_script = f"""
import json, os, sys, time
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")
from tq.server import load_state, stop_server, _check_health, _validate_pid, _clear_state

STATE_FILE = "{STATE_FILE}"
WATCHER_PID_FILE = "{WATCHER_PID_FILE}"
IDLE_TIMEOUT = {state.idle_timeout}
PORT = {state.port}
HOST = "{state.host}"
PID = {state.pid}
BINARY = "{state.binary_path}"

with open(WATCHER_PID_FILE, "w") as f:
    f.write(str(os.getpid()))

last_active = time.time()

while True:
    try:
        if not _validate_pid(PID, BINARY):
            break
        if _check_health(HOST, PORT):
            try:
                import urllib.request
                url = f"http://{{HOST}}:{{PORT}}/v1/models"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=3) as resp:
                    last_active = time.time()
            except Exception:
                import urllib.request
                try:
                    url = f"http://{{HOST}}:{{PORT}}/health"
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        pass
                except Exception:
                    pass
        else:
            break

        idle = time.time() - last_active
        if idle >= IDLE_TIMEOUT:
            stop_server()
            break

        time.sleep(5)
    except Exception:
        break

if os.path.isfile(WATCHER_PID_FILE):
    os.unlink(WATCHER_PID_FILE)
"""

    subprocess.Popen(
        [sys.executable, "-c", watcher_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _stop_watcher() -> None:
    if not os.path.isfile(WATCHER_PID_FILE):
        return
    try:
        with open(WATCHER_PID_FILE, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 9)
    except (ValueError, ProcessLookupError, PermissionError):
        pass
    finally:
        if os.path.isfile(WATCHER_PID_FILE):
            os.unlink(WATCHER_PID_FILE)
