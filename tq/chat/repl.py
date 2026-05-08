from __future__ import annotations

from .client import ChatClient, ChatMessage, ToolCall
from .tools import ToolRegistry, ToolResult
from .permissions import PermissionConfig, PermissionAction
from .render import (
    console, render_markdown, render_streaming, render_tool_call,
    render_tool_result, render_status, render_error, render_info, render_divider,
)
from ..server import get_server_status
from ..scanner import scan_models, resolve_model_path
from ..parser import build_model_metadata
from ..hardware import detect_hardware
from ..recommender import recommend
from .. import config as cfg
from ..types import ServerConfig

import os
import sys
import threading


class ChatSession:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "",
        workdir: str = "",
        system_prompt: str = "",
        perms: PermissionConfig | None = None,
    ):
        self.client = ChatClient(base_url=base_url)
        self.base_url = base_url
        self.model = model
        self.workdir = os.path.abspath(workdir or os.getcwd())
        self.system_prompt = system_prompt
        self.perms = perms or PermissionConfig.defaults()
        self.messages: list[ChatMessage] = []
        self.tools = ToolRegistry(self.workdir, self.perms)
        self.cancel_event = threading.Event()

    def start(self):
        if not self.client.is_available():
            render_error(f"No server running at {self.base_url}")
            render_info("Run 'tq serve' first to start a model server, then 'tq chat'")
            return

        models = self.client.list_models()
        if models:
            self.model = self.model or models[0].get("id", "")
        else:
            state = load_state()
            if state:
                self.model = os.path.basename(state.model_path)

        if not self.model:
            render_error("No model available")
            return

        if self.system_prompt:
            self.messages.append(ChatMessage(role="system", content=self.system_prompt))
        else:
            self.messages.append(ChatMessage(
                role="system",
                content=self._default_system_prompt(),
            ))

        render_status(self.model, self.base_url)
        render_info("Type /help for commands. Ctrl+C to cancel, Ctrl+D to quit.")
        render_divider()

        self._repl_loop()

    def _repl_loop(self):
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
            hist_dir = os.path.expanduser("~/.tq")
            os.makedirs(hist_dir, exist_ok=True)
            prompt_session = PromptSession(
                history=FileHistory(os.path.join(hist_dir, "chat_history")),
            )
            use_prompt_toolkit = True
        except ImportError:
            use_prompt_toolkit = False

        while True:
            try:
                if use_prompt_toolkit:
                    user_input = prompt_session.prompt("  You: ").strip()
                else:
                    user_input = input("  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    break
                continue

            self.messages.append(ChatMessage(role="user", content=user_input))
            self.cancel_event.clear()
            self._send_and_process()

    def _send_and_process(self):
        max_iterations = 20
        for _ in range(max_iterations):
            response_text = ""
            tool_calls: list[ToolCall] = []
            current_tc: dict = {}

            render_divider()

            try:
                for chunk in self.client.stream_chat(
                    self.messages,
                    model=self.model,
                    tools=self.tools.get_tool_definitions(),
                ):
                    if self.cancel_event.is_set():
                        break

                    if chunk.delta_content:
                        response_text += chunk.delta_content
                        print(chunk.delta_content, end="", flush=True)

                    if chunk.tool_call_id:
                        if "id" not in current_tc:
                            current_tc = {"id": chunk.tool_call_id, "name": "", "args": ""}
                        if chunk.tool_call_name:
                            current_tc["name"] = chunk.tool_call_name
                        if chunk.tool_call_args:
                            current_tc["args"] += chunk.tool_call_args

                    if chunk.finish_reason == "tool_calls" and current_tc:
                        tool_calls.append(ToolCall(
                            id=current_tc["id"],
                            name=current_tc["name"],
                            arguments=current_tc["args"],
                        ))
                        current_tc = {}

                    if chunk.finish_reason == "stop":
                        if current_tc.get("id"):
                            tool_calls.append(ToolCall(
                                id=current_tc["id"],
                                name=current_tc["name"],
                                arguments=current_tc["args"],
                            ))
                        current_tc = {}

            except KeyboardInterrupt:
                print(flush=True)
                render_info("Generation cancelled")
                if response_text:
                    self.messages.append(ChatMessage(role="assistant", content=response_text))
                break
            except Exception as e:
                print(flush=True)
                render_error(f"Request failed: {e}")
                break

            if response_text:
                print(flush=True)

            if not tool_calls:
                if response_text:
                    self.messages.append(ChatMessage(role="assistant", content=response_text))
                break

            self.messages.append(ChatMessage(role="assistant", content=response_text, tool_calls=tool_calls))

            for tc in tool_calls:
                args = tc.parsed_args()
                render_tool_call(tc.name, args)
                result = self.tools.execute(tc)
                render_tool_result(tc.name, result.output, result.error)

                self.messages.append(ChatMessage(
                    role="tool",
                    content=result.output,
                    tool_call_id=tc.id,
                    name=tc.name,
                ))

        token_count = sum(len(m.content) for m in self.messages) // 4
        print(f"  [dim]{token_count:,} tokens in context[/dim]", flush=True)

    def _handle_command(self, cmd: str) -> bool:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ("/quit", "/exit", "/q"):
            return True
        elif command == "/help":
            self._cmd_help()
        elif command == "/clear":
            self.messages = [m for m in self.messages if m.role == "system"]
            render_info("Conversation cleared")
        elif command == "/compact":
            self._cmd_compact()
        elif command == "/model":
            self._cmd_model(arg)
        elif command == "/status":
            self._cmd_status()
        elif command == "/tools":
            self._cmd_tools()
        elif command == "/undo":
            self._cmd_undo()
        elif command == "/copy":
            self._cmd_copy()
        elif command == "/retry":
            self._cmd_retry()
        else:
            render_error(f"Unknown command: {command}")
        return False

    def _cmd_help(self):
        console.print("""
  [bold]Commands:[/bold]
    /help       Show this help
    /clear      Clear conversation history
    /compact    Summarize and compress context
    /model      List or switch models
    /status     Show server and session status
    /tools      List available tools
    /undo       Remove last exchange
    /copy       Copy last response to clipboard
    /retry      Regenerate last response
    /quit       Exit chat (Ctrl+D)

  [bold]Keys:[/bold]
    Ctrl+C      Cancel current generation
    Ctrl+D      Exit chat
""")

    def _cmd_compact(self):
        render_info("Compressing conversation...")
        summary_msg = ChatMessage(
            role="user",
            content="Summarize our entire conversation so far in a concise paragraph. Include key decisions, files discussed, and current state. This summary will replace the conversation history.",
        )
        summary = ""
        try:
            for chunk in self.client.stream_chat(
                self.messages + [summary_msg],
                model=self.model,
            ):
                if chunk.delta_content:
                    summary += chunk.delta_content
                    print(chunk.delta_content, end="", flush=True)
        except Exception as e:
            render_error(f"Compaction failed: {e}")
            return

        print(flush=True)
        if summary:
            self.messages = [
                self.messages[0],
                ChatMessage(role="assistant", content=f"[Previous conversation summary: {summary}]"),
            ]
            render_info("Context compressed")

    def _cmd_model(self, arg: str):
        models = self.client.list_models()
        if not models:
            render_info("No models available on server")
            return

        if arg:
            for m in models:
                if arg.lower() in m.get("id", "").lower():
                    self.model = m["id"]
                    render_info(f"Switched to {self.model}")
                    return
            render_error(f"Model not found: {arg}")
            return

        console.print("  [bold]Available models:[/bold]")
        for m in models:
            mid = m.get("id", "?")
            marker = "●" if mid == self.model else "○"
            console.print(f"    {marker} {mid}")

    def _cmd_status(self):
        status = get_server_status()
        if status:
            console.print(f"  Server: PID {status['pid']}, {status['host']}:{status['port']}")
            console.print(f"  Model: {status['model']}")
            console.print(f"  Healthy: {status['healthy']}")
        else:
            render_info("No server running")

        msg_count = len(self.messages)
        token_est = sum(len(m.content) for m in self.messages) // 4
        console.print(f"  Messages: {msg_count}, ~{token_est:,} tokens")

    def _cmd_tools(self):
        console.print("  [bold]Available tools:[/bold]")
        for name, defn in self.tools.get_tool_definitions():
            desc = defn.get("function", {}).get("description", "")
            console.print(f"    {name:12s} {desc}")

    def _cmd_undo(self):
        while self.messages and self.messages[-1].role != "user":
            self.messages.pop()
        if self.messages and self.messages[-1].role == "user":
            self.messages.pop()
        render_info("Removed last exchange")

    def _cmd_copy(self):
        last_assistant = None
        for m in reversed(self.messages):
            if m.role == "assistant" and m.content:
                last_assistant = m.content
                break
        if not last_assistant:
            render_info("No assistant response to copy")
            return
        try:
            import subprocess
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(last_assistant.encode())
            render_info("Copied to clipboard")
        except Exception:
            try:
                import subprocess
                proc = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
                proc.communicate(last_assistant.encode())
                render_info("Copied to clipboard")
            except Exception:
                render_info("Clipboard not available")

    def _cmd_retry(self):
        self._cmd_undo()
        if self.messages and self.messages[-1].role == "user":
            user_msg = self.messages.pop()
            self.messages.append(user_msg)
            self._send_and_process()

    def _default_system_prompt(self) -> str:
        return (
            "You are an AI coding assistant running locally via tq chat. "
            "You have access to tools for reading, writing, editing files, "
            "running bash commands, searching the web, and more. "
            "Be concise and direct. Use tools when needed to accomplish tasks. "
            "If a tool call fails or returns an error, report the error to the user and move on. "
            "Do not retry failed tool calls. "
            f"Working directory: {self.workdir}"
        )
