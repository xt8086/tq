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


_HELPER_CALL_RE = re.compile(
    r'(?:^|\n)\s*(websearch|curl|weather|exec)\s*\(\s*(["\'].*?["\'])\s*\)',
    re.MULTILINE,
)

_BARE_CODE_RE = re.compile(
    r'^\s*(?:from\s+\w|import\s+\w|def\s+\w|class\s+\w|if\s+|for\s+|while\s+|try:|with\s+|'
    r'\w+\s*=\s*(?:weather|curl|websearch|subprocess)\s*\(|'
    r'print\s*\(|subprocess\.run)',
    re.MULTILINE,
)


_SHELL_CMD_RE = re.compile(
    r'^\s*(?:ifconfig|ipconfig|netstat|ping|traceroute|whoami|hostname|uname|'
    r'ls|cat|head|tail|grep|find|ps|df|du|top|kill|systemctl|networksetup|'
    r'scutil|defaults|diskutil|brew|port|npm|pip|which|whereis|echo|mkdir|'
    r'cp|mv|rm|chmod|chown|curl|wget|dig|nslookup|ssh|scp|rsync|git|docker|'
    r'python3?|node|ruby|java|gcc|make|cmake|cargo)\b',
    re.MULTILINE,
)


def extract_python_blocks(text: str) -> list[str]:
    exec_blocks = re.findall(r'```exec\s*\n(.*?)```', text, re.DOTALL)
    if exec_blocks:
        return [_wrap_if_shell(b) for b in exec_blocks]
    blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks
    bash_blocks = re.findall(r'```bash\s*\n(.*?)```', text, re.DOTALL)
    if bash_blocks:
        return [_wrap_if_shell(b) for b in bash_blocks]
    helper_calls = _HELPER_CALL_RE.findall(text)
    if helper_calls:
        code_parts = []
        for func, arg in helper_calls:
            if func == 'exec':
                code_parts.append(arg.strip('"\''))
            else:
                code_parts.append(f'print({func}({arg}))')
        if code_parts:
            return ['\n'.join(code_parts)]
    lines = text.splitlines()
    bare = []
    for line in lines:
        if _BARE_CODE_RE.match(line):
            bare.append(line)
        elif bare and line.strip() and not line.strip().startswith(('- ', '* ', '# ', '1.', '2.', '3.')):
            bare.append(line)
        elif bare:
            break
    if bare:
        return ['\n'.join(bare)]
    return []


def _wrap_if_shell(code: str) -> str:
    stripped = code.strip()
    if '\n' not in stripped and _SHELL_CMD_RE.match(stripped):
        return f'print(exec({repr(stripped)}))'
    lines = stripped.splitlines()
    all_shell = all(not l.strip() or _SHELL_CMD_RE.match(l.strip()) or l.strip().startswith('|') or l.strip().startswith('>') or l.strip().startswith(';') or l.strip().startswith('#') or l.strip().startswith('-') for l in lines)
    if all_shell and any(_SHELL_CMD_RE.match(l.strip()) for l in lines if l.strip() and not l.strip().startswith('#')):
        parts = []
        for l in lines:
            s = l.strip()
            if not s or s.startswith('#'):
                continue
            parts.append(f'print(exec({repr(s)}))')
        return '\n'.join(parts)
    return code


_AUTO_IMPORTS = """import subprocess,os,json,sys,math,re,datetime,pathlib,urllib.parse,urllib.request
def curl(url,timeout=10):
 if not url.startswith('http'): url='https://'+url
 return subprocess.run(['curl','-sL',url],capture_output=True,text=True,timeout=timeout).stdout
def weather(location):
 loc=urllib.parse.quote(location.replace(' ','+'))
 r=curl(f'wttr.in/{loc}?format=j1');d=json.loads(r);a=d['nearest_area'][0];c=d['current_condition'][0];return {'area':a['areaName'][0]['value'],'region':a['region'][0]['value'],'temp_F':c['temp_F'],'feels_F':c['FeelsLikeF'],'humidity':c['humidity'],'desc':c['weatherDesc'][0]['value'],'wind_mph':c['windspeedMiles']}
def websearch(query,num=3):
 r=subprocess.run(['curl','-s','-X','POST','https://mcp.exa.ai/mcp','-H','Content-Type: application/json','-H','Accept: application/json, text/event-stream','-d',json.dumps({"jsonrpc":"2.0","method":"tools/call","params":{"name":"web_search_exa","arguments":{"query":query,"numResults":num}},"id":1})],capture_output=True,text=True,timeout=15).stdout;lines=[l for l in r.strip().splitlines() if l.startswith('data:')];data=[json.loads(l[5:]) for l in lines if l[5:].strip()];raw=[];[raw.extend(c.get('result',{}).get('content',[])) for c in data if 'result' in c];texts=[item['text'] if isinstance(item,dict) and 'text' in item else str(item) for item in raw];return '\\n---\\n'.join(texts) if texts else 'No results found'
def exec(cmd,timeout=10):
 return subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=timeout).stdout
"""

def _smart_truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit // 2
    return text[:head] + f"\n\n... ({len(text) - limit:,} chars omitted) ...\n\n" + text[-tail:]


def execute_python_code(code: str, workdir: str, timeout: int = 30) -> tuple[str, bool]:
    full_code = _AUTO_IMPORTS + code
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        output = result.stdout
        if result.stderr:
            stderr = "\n".join(
                line for line in result.stderr.strip().splitlines()
                if "MallocStackLogging" not in line
            )
            if stderr:
                output += ("\n" if output else "") + stderr
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
        self._is_mm = False
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

        self._is_mm = bool(state and state.is_multimodal) if state else False

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

            first_token = user_input.split()[0]
            if user_input.startswith("/") and not (os.path.sep in first_token[1:] or "." in first_token):
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
                if "400" in str(e):
                    self._trim_oversized_messages()
                    render_info("Trimmed oversized messages — retrying...")
                    continue
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
            output = _smart_truncate(output, 4000)
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
            if "400" in str(e):
                self._trim_oversized_messages()
                render_info("Trimmed oversized messages — retrying...")
                self._send_and_process_no_tools()
            else:
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

    def _trim_oversized_messages(self):
        for i in range(len(self.messages) - 1, -1, -1):
            m = self.messages[i]
            if len(m.content) > 4000:
                m.content = _smart_truncate(m.content, 4000)
                return
        if len(self.messages) > 4:
            del self.messages[1:-2]

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
                        console.print(f"  [dim]Attached PDF (vision): {raw_path} ({len(images)} page(s))[/dim]")
                text_content = self._extract_pdf_text(expanded)
                if text_content:
                    msg.content += f"\n\n[File content: {raw_path}]\n{text_content[:30000]}"
                    console.print(f"  [dim]Attached PDF (text): {raw_path}[/dim]")
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
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text.strip())
            if pages:
                return "\n\n".join(pages)[:30000]
        except ImportError:
            pass
        except Exception:
            pass
        try:
            import subprocess
            result = subprocess.run(
                ["pdftotext", path, "-"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()[:30000]
        except FileNotFoundError:
            pass
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
            "Instead, to perform actions just output a helper call directly — no code blocks, no Python syntax needed:\n"
            "- To search the web: websearch(\"your search query\")\n"
            "- To fetch a URL: curl(\"https://example.com\")\n"
            "- To get weather: weather(\"92880\") or weather(\"Corona, CA\")\n"
            "- To run a shell command: exec(\"ls -la\") or exec(\"ifconfig\")\n"
            "\nJust write the call with your argument — the system handles the rest.\n"
            "\nExamples:\n"
            '  User: weather in Corona CA → weather("Corona, CA")\n'
            '  User: search for AI news → websearch("AI news today")\n'
            '  User: what is on that page → curl("https://example.com")\n'
            '  User: list files → exec("ls -la")\n'
            '  User: network info → exec("ifconfig")\n'
            '  User: am I online → exec("ping -c 1 google.com")\n'
            "\nRULES:\n"
            '1. Always use quotes around your argument: weather("92880") not weather(92880)\n'
            "2. For multi-step tasks, output multiple calls on separate lines\n"
            "3. websearch() is best for questions and lookups\n"
            "4. weather() gives current conditions only; use websearch() or curl() for forecasts\n"
            "5. curl() is for fetching a known URL\n"
            "6. exec() is for ANY shell command — ifconfig, netstat, ping, ls, git, etc.\n"
            "7. NEVER use mock data or make up results — only report what the calls return\n"
            "8. If you also know Python, you can write ```exec code blocks for complex logic\n"
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
        return self._is_mm
