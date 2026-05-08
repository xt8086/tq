from __future__ import annotations

import re

from .client import ChatClient, ChatMessage, ToolCall, load_image_as_base64, load_pdf_as_images, load_docx_as_text, load_xlsx_as_text, load_csv_as_text
from .tools import ToolRegistry, ToolResult
from .permissions import PermissionConfig, PermissionAction
from .render import (
    console, render_markdown, render_streaming, render_tool_call,
    render_tool_result, render_status, render_error, render_info, render_divider,
)
from ..server import get_server_status, load_state
from ..scanner import scan_models, resolve_model_path
from ..parser import build_model_metadata
from ..hardware import detect_hardware
from ..recommender import recommend
from .. import config as cfg
from ..types import ServerConfig

import os
import subprocess
import sys
import threading


def extract_python_blocks(text: str) -> list[str]:
    exec_blocks = re.findall(r'```exec\s*\n(.*?)```', text, re.DOTALL)
    if exec_blocks:
        return exec_blocks
    blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks
    return re.findall(r'```bash\s*\n(.*?)```', text, re.DOTALL)


def execute_python_code(code: str, workdir: str, timeout: int = 30) -> tuple[str, bool]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.returncode != 0 and not result.stderr:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip(), result.returncode != 0
    except subprocess.TimeoutExpired:
        return "Execution timed out", True
    except Exception as e:
        return str(e), True


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
        self.tools_disabled = False
        self._code_exec_count = 0
        self._code_exec_limit = 5
        self.cancel_event = threading.Event()

    def start(self):
        if not self.client.is_available():
            render_error(f"No server running at {self.base_url}")
            render_info("Run 'tq serve' first to start a model server, then 'tq chat'")
            return

        state = load_state()
        tool_support = state.tool_support if state else "none"

        models = self.client.list_models()
        if models:
            self.model = self.model or models[0].get("id", "")
        elif state:
            self.model = os.path.basename(state.model_path)

        if not self.model:
            render_error("No model available")
            return

        if tool_support == "none":
            self.tools_disabled = True

        if self.system_prompt:
            self.messages.append(ChatMessage(role="system", content=self.system_prompt))
        else:
            prompt = self._default_system_prompt()
            if self.tools_disabled:
                prompt += self._code_block_system_addendum()
            self.messages.append(ChatMessage(role="system", content=prompt))

        mode_label = "code blocks" if self.tools_disabled else "tool calling"
        render_status(self.model, self.base_url, f"Mode: {mode_label}")
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

            msg = ChatMessage(role="user", content=user_input)
            if self._is_multimodal():
                msg = self._attach_files(user_input, msg)
            self.messages.append(msg)
            self.cancel_event.clear()
            self._code_exec_count = 0
            self._send_and_process()

    def _send_and_process(self):
        max_iterations = 20
        for _ in range(max_iterations):
            response_text = ""
            tool_calls: list[ToolCall] = []
            current_tc: dict = {}
            msg_count_before = len(self.messages)

            render_divider()

            try:
                for chunk in self.client.stream_chat(
                    self.messages,
                    model=self.model,
                    tools=None if self.tools_disabled else self.tools.get_tool_definitions(),
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
                if msg_count_before < len(self.messages):
                    del self.messages[msg_count_before:]
                if "500" in str(e) and tool_calls:
                    render_info("Model may not support tool calling — retrying without tools")
                    self._send_without_tools()
                break

            if response_text:
                print(flush=True)

            if not tool_calls:
                if response_text:
                    self.messages.append(ChatMessage(role="assistant", content=response_text))
                    if self.tools_disabled:
                        self._process_code_blocks(response_text)
                break

            empty_args = []
            valid_tool_calls = []
            for tc in tool_calls:
                args = tc.parsed_args()
                if not args:
                    empty_args.append(tc.name)
                    continue
                valid_tool_calls.append(tc)

            if empty_args:
                render_error(f"{', '.join(empty_args)}() called with no/empty arguments — model may not support tool calling")
                if not self.tools_disabled:
                    self.tools_disabled = True
                    render_info("Switching to code block execution mode — retrying...")
                    if self.messages and self.messages[0].role == "system":
                        self.messages[0].content += self._code_block_system_addendum()
                    if msg_count_before < len(self.messages):
                        del self.messages[msg_count_before:]
                    self._send_and_process_no_tools()
                break

            self.messages.append(ChatMessage(role="assistant", content=response_text, tool_calls=valid_tool_calls))

            tool_results = []
            for tc in valid_tool_calls:
                args = tc.parsed_args()
                render_tool_call(tc.name, args)
                result = self.tools.execute(tc)
                render_tool_result(tc.name, result.output, result.error)
                tool_results.append((tc, result))

            for tc, result in tool_results:
                self.messages.append(ChatMessage(
                    role="user",
                    content=f"[Tool result for {tc.name}]: {result.output}" + (" (ERROR)" if result.error else ""),
                ))

        token_count = sum(len(m.content) for m in self.messages) // 4
        console.print(f"  [dim]{token_count:,} tokens in context[/dim]")

    def _process_code_blocks(self, text: str):
        blocks = extract_python_blocks(text)
        if not blocks:
            return

        if self._code_exec_count >= self._code_exec_limit:
            render_info(f"Code execution limit reached ({self._code_exec_limit}) — stopping to prevent loop")
            return

        for code in blocks:
            self._code_exec_count += 1
            preview = code.strip().split("\n")[0][:80]
            console.print(f"  [cyan]→ python: {preview}[/cyan]")

            action = self.perms.get_action("bash", code, self.workdir)
            if action == PermissionAction.ASK:
                action = ask_permission("python", code.strip()[:200])

            if action == PermissionAction.DENY:
                console.print("  [dim]⊘ Skipped (denied)[/dim]")
                self.messages.append(ChatMessage(
                    role="user",
                    content="[Code execution was denied by user]",
                ))
                continue

            output, is_error = execute_python_code(code, self.workdir)
            if output:
                preview_out = output[:300] + ("..." if len(output) > 300 else "")
                if is_error:
                    console.print(f"  [bold red]✗ python:[/bold red] {preview_out}")
                else:
                    console.print(f"  [dim]← {preview_out}[/dim]")

            self.messages.append(ChatMessage(
                role="user",
                content=f"[Code output]: {output}" + (" (ERROR)" if is_error else ""),
            ))

            if self._code_exec_count >= self._code_exec_limit:
                render_info(f"Code execution limit reached ({self._code_exec_limit})")
                return

            self.cancel_event.clear()
            self._send_and_process_no_tools()
            break

    def _send_and_process_no_tools(self):
        try:
            response_text = ""
            for chunk in self.client.stream_chat(
                self.messages,
                model=self.model,
                tools=None,
            ):
                if self.cancel_event.is_set():
                    break
                if chunk.delta_content:
                    response_text += chunk.delta_content
                    print(chunk.delta_content, end="", flush=True)
            print(flush=True)
            if response_text:
                self.messages.append(ChatMessage(role="assistant", content=response_text))
                self._process_code_blocks(response_text)
        except Exception as e:
            print(flush=True)
            render_error(f"Request failed: {e}")

    def _send_without_tools(self):
        try:
            response_text = ""
            for chunk in self.client.stream_chat(
                self.messages,
                model=self.model,
                tools=None,
            ):
                if chunk.delta_content:
                    response_text += chunk.delta_content
                    print(chunk.delta_content, end="", flush=True)
            print(flush=True)
            if response_text:
                self.messages.append(ChatMessage(role="assistant", content=response_text))
        except Exception as e:
            print(flush=True)
            render_error(f"Retry failed: {e}")

    def _attach_files(self, text: str, msg: ChatMessage) -> ChatMessage:
        import re as _re
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
        text_doc_exts = {".docx", ".xlsx", ".xls", ".csv", ".tsv", ".txt", ".md", ".json", ".xml", ".yaml", ".yml", ".log", ".py", ".js", ".ts", ".html", ".css"}
        paths = _re.findall(r'(?:[\w./~-]+/[\w.-]+\.\w+|~/[\w./-]+\.\w+)', text)
        for raw_path in paths:
            expanded = os.path.expanduser(raw_path)
            if not os.path.isfile(expanded):
                continue
            ext = os.path.splitext(expanded)[1].lower()
            size_mb = os.path.getsize(expanded) / (1024 * 1024)
            if ext == ".pdf":
                if self._is_multimodal():
                    images = load_pdf_as_images(expanded)
                    if images:
                        msg.images.extend(images)
                        console.print(f"  [dim]Attached PDF: {raw_path} ({len(images)} page(s))[/dim]")
                text_content = self._extract_pdf_text(expanded)
                if text_content:
                    msg.content += f"\n\n[File content: {raw_path}]\n{text_content[:30000]}"
            elif ext in image_exts:
                if self._is_multimodal():
                    img = load_image_as_base64(expanded)
                    if img:
                        msg.images.append(img)
                        console.print(f"  [dim]Attached image: {raw_path}[/dim]")
            elif ext == ".docx":
                content = load_docx_as_text(expanded)
                if content:
                    msg.content += f"\n\n[File content: {raw_path}]\n{content[:30000]}"
                    console.print(f"  [dim]Attached doc: {raw_path}[/dim]")
            elif ext in (".xlsx", ".xls"):
                content = load_xlsx_as_text(expanded)
                if content:
                    msg.content += f"\n\n[File content: {raw_path}]\n{content[:30000]}"
                    console.print(f"  [dim]Attached spreadsheet: {raw_path}[/dim]")
            elif ext in (".csv", ".tsv"):
                content = load_csv_as_text(expanded)
                if content:
                    msg.content += f"\n\n[File content: {raw_path}]\n{content[:30000]}"
                    console.print(f"  [dim]Attached CSV: {raw_path}[/dim]")
            elif ext in text_doc_exts and size_mb < 1:
                try:
                    with open(expanded, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read(30000)
                    msg.content += f"\n\n[File content: {raw_path}]\n{content}"
                    console.print(f"  [dim]Attached file: {raw_path}[/dim]")
                except Exception:
                    pass
        return msg

    def _extract_pdf_text(self, path: str) -> Optional[str]:
        try:
            import subprocess
            result = subprocess.run(
                ["pdftotext", path, "-"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()[:30000]
        except Exception:
            pass
        return None

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

    def _code_block_system_addendum(self) -> str:
        base = (
            "\n\nIMPORTANT: You do NOT have access to tool-calling. "
            "Instead, when you need to execute commands or perform actions, "
            "write Python code inside ```exec code blocks. "
            "Your code will be automatically extracted, executed, and the output fed back to you. "
            "Only write code you actually need executed — do not write example or illustrative code blocks. "
            "\n\nRULES:\n"
            "1. Use ```exec for code you want executed. Use ```python only for examples you do NOT want executed.\n"
            "2. For shell commands: subprocess.run(['cmd', 'arg'], capture_output=True, text=True)\n"
            "3. For HTTP requests: use subprocess.run(['curl', '-s', url], capture_output=True, text=True) — this works without any API key or library.\n"
            "4. For web search: subprocess.run(['curl', '-s', '-X', 'POST', 'https://mcp.exa.ai/mcp', '-H', 'Content-Type: application/json', '-H', 'Accept: application/json, text/event-stream', '-d', json.dumps({\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"params\":{\"name\":\"web_search_exa\",\"arguments\":{\"query\":\"YOUR QUERY\",\"numResults\":3}},\"id\":1})], capture_output=True, text=True)\n"
            "5. NEVER use mock data, placeholder responses, simulated output, or API keys like 'YOUR_API_KEY'.\n"
            "6. NEVER import requests — it is not installed. Use curl via subprocess instead.\n"
            "7. Always print your results so they appear in the output.\n"
            "\n\nFREE APIs (no key needed, use curl):\n"
            "- Weather: subprocess.run(['curl', '-s', 'wttr.in/92880?format=3'], capture_output=True, text=True)\n"
            "- Weather JSON: subprocess.run(['curl', '-s', 'wttr.in/92880?format=j1'], capture_output=True, text=True)\n"
            "- IP info: subprocess.run(['curl', '-s', 'ifconfig.me'], capture_output=True, text=True)\n"
            "- Web fetch: subprocess.run(['curl', '-s', '-L', url], capture_output=True, text=True)\n"
            "- Web search: subprocess.run(['curl', '-s', '-X', 'POST', 'https://mcp.exa.ai/mcp', '-H', 'Content-Type: application/json', '-H', 'Accept: application/json, text/event-stream', '-d', json.dumps({\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"params\":{\"name\":\"web_search_exa\",\"arguments\":{\"query\":\"YOUR QUERY\",\"numResults\":3}},\"id\":1})], capture_output=True, text=True)\n"
        )
        if self._is_multimodal():
            base += (
                "\n\nVISION: You have multimodal/vision capabilities. "
                "When the user asks about an image or document, it will be attached automatically for you to see and analyze. "
                "You do NOT need to extract text or use subprocess to read images/PDFs.\n"
            )
        base += f"\nWorking directory: {self.workdir}"
        return base

    def _is_multimodal(self) -> bool:
        state = load_state()
        return state is not None and state.is_multimodal
