from __future__ import annotations

import fnmatch
import glob as globmod
import json
import os
import re
import subprocess
import tempfile
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

from .client import ToolCall
from .permissions import PermissionAction, PermissionConfig, ask_permission


@dataclass
class ToolResult:
    output: str
    error: bool = False
    display: str = ""


class ToolRegistry:
    def __init__(self, workdir: str, perms: PermissionConfig):
        self.workdir = os.path.abspath(workdir)
        self.perms = perms
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, callable] = {}
        self._doom_counter: dict[str, int] = {}
        self._register_all()

    def get_tool_definitions(self) -> list[dict]:
        return list(self._tools.values())

    def execute(self, tool_call: ToolCall) -> ToolResult:
        name = tool_call.name
        args = tool_call.parsed_args()

        handler = self._handlers.get(name)
        if not handler:
            return ToolResult(f"Unknown tool: {name}", error=True)

        doom_key = f"{name}:{tool_call.arguments}"
        self._doom_counter[doom_key] = self._doom_counter.get(doom_key, 0) + 1
        if self._doom_counter[doom_key] >= 3:
            action = self.perms.get_action("doom_loop", doom_key)
            if action == PermissionAction.ASK:
                result = ask_permission("doom_loop", f"{name} repeated 3 times")
                if result == PermissionAction.DENY:
                    return ToolResult("Rejected: tool call repeated 3 times with identical input", error=True)
            elif action == PermissionAction.DENY:
                return ToolResult("Rejected: tool call repeated 3 times", error=True)

        input_val = self._get_input_for_perm(name, args)
        action = self.perms.get_action(name, input_val, self.workdir)

        if action == PermissionAction.DENY:
            return ToolResult(f"Permission denied for {name}", error=True)
        if action == PermissionAction.ASK:
            result = ask_permission(name, input_val)
            if result == PermissionAction.DENY:
                return ToolResult(f"Permission denied for {name}", error=True)

        try:
            return handler(args)
        except Exception as e:
            return ToolResult(f"Tool error ({name}): {e}", error=True)

    def _get_input_for_perm(self, tool: str, args: dict) -> str:
        if tool == "bash":
            return args.get("command", "")
        if tool in ("edit", "write", "read"):
            return args.get("file_path", args.get("path", ""))
        if tool in ("glob", "grep"):
            return args.get("pattern", args.get("include", ""))
        if tool == "webfetch":
            return args.get("url", "")
        if tool == "websearch":
            return args.get("query", "")
        if tool == "apply_patch":
            return args.get("patchText", "")
        return ""

    def _register_all(self):
        self._reg("bash", self._tool_bash,
                  "Execute a bash command in the working directory",
                  {"command": {"type": "string", "description": "The bash command to execute"},
                   "workdir": {"type": "string", "description": "Working directory (default: project root)"}})
        self._reg("read", self._tool_read,
                  "Read a file from the filesystem",
                  {"file_path": {"type": "string", "description": "Absolute path to the file"},
                   "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
                   "limit": {"type": "integer", "description": "Maximum number of lines to read"}})
        self._reg("edit", self._tool_edit,
                  "Perform an exact string replacement in a file",
                  {"file_path": {"type": "string", "description": "Absolute path to the file"},
                   "oldString": {"type": "string", "description": "The text to replace"},
                   "newString": {"type": "string", "description": "The replacement text"},
                   "replaceAll": {"type": "boolean", "description": "Replace all occurrences"}})
        self._reg("write", self._tool_write,
                  "Write content to a file, creating or overwriting",
                  {"file_path": {"type": "string", "description": "Absolute path to the file"},
                   "content": {"type": "string", "description": "Content to write"}})
        self._reg("glob", self._tool_glob,
                  "Find files matching a glob pattern",
                  {"pattern": {"type": "string", "description": "Glob pattern (e.g. **/*.py)"},
                   "path": {"type": "string", "description": "Directory to search in"}})
        self._reg("grep", self._tool_grep,
                  "Search file contents using regex",
                  {"pattern": {"type": "string", "description": "Regex pattern to search for"},
                   "path": {"type": "string", "description": "Directory to search in"},
                   "include": {"type": "string", "description": "File pattern to include (e.g. *.py)"}})
        self._reg("apply_patch", self._tool_apply_patch,
                  "Apply a patch to files",
                  {"patchText": {"type": "string", "description": "Patch text with file markers"}})
        self._reg("webfetch", self._tool_webfetch,
                  "Fetch content from a URL",
                  {"url": {"type": "string", "description": "URL to fetch"},
                   "format": {"type": "string", "enum": ["text", "markdown", "html"], "description": "Format to return"}})
        self._reg("websearch", self._tool_websearch,
                  "Search the web for information",
                  {"query": {"type": "string", "description": "Search query"}})
        self._reg("question", self._tool_question,
                  "Ask the user a question during execution",
                  {"question": {"type": "string", "description": "The question to ask"},
                   "header": {"type": "string", "description": "Short label for the question"}})
        self._reg("todowrite", self._tool_todowrite,
                  "Create or update a todo list",
                  {"todos": {"type": "array", "description": "List of todo items with content, status, priority"}})

    def _reg(self, name: str, handler: callable, description: str, params: dict):
        properties = {}
        required = []
        for pname, pdef in params.items():
            properties[pname] = pdef
            if pdef.get("type") != "boolean" and "default" not in pdef:
                required.append(pname)
        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        self._handlers[name] = handler

    def _tool_bash(self, args: dict) -> ToolResult:
        command = args.get("command", "")
        workdir = args.get("workdir", self.workdir)
        if not command:
            return ToolResult("No command provided", error=True)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout
            if result.stderr:
                output += f"[stderr] {result.stderr}" if output else result.stderr
            if not output:
                output = f"(exit code: {result.returncode})"
            elif result.returncode != 0:
                output += f"(exit code: {result.returncode})"
            return ToolResult(output, error=result.returncode != 0)
        except subprocess.TimeoutExpired:
            return ToolResult("Command timed out after 120s", error=True)

    def _tool_read(self, args: dict) -> ToolResult:
        file_path = args.get("file_path", args.get("path", ""))
        if not file_path:
            return ToolResult("No file path provided", error=True)

        if not os.path.isfile(file_path):
            return ToolResult(f"File not found: {file_path}", error=True)

        offset = args.get("offset", 0)
        limit = args.get("limit", 2000)

        try:
            with open(file_path, errors="replace") as f:
                lines = f.readlines()

            if offset:
                lines = lines[offset - 1:]
            if limit:
                lines = lines[:limit]

            numbered = []
            for i, line in enumerate(lines, start=(offset or 1)):
                stripped = line.rstrip()
                if len(stripped) > 2000:
                    stripped = stripped[:2000] + "..."
                numbered.append(f"{i}: {stripped}")

            result = f"File: {file_path} ({len(lines)} lines shown"
            if offset or limit != 2000:
                result += f", offset={offset}, limit={limit}"
            result += "):" + "".join(f"  {num}  " for num in numbered[:1])
            result = "".join(numbered)

            return ToolResult(result)
        except Exception as e:
            return ToolResult(f"Error reading file: {e}", error=True)

    def _tool_edit(self, args: dict) -> ToolResult:
        file_path = args.get("file_path", "")
        old_string = args.get("oldString", "")
        new_string = args.get("newString", "")
        replace_all = args.get("replaceAll", False)

        if not file_path or not old_string:
            return ToolResult("file_path and oldString are required", error=True)

        if not os.path.isfile(file_path):
            return ToolResult(f"File not found: {file_path}", error=True)

        try:
            with open(file_path, "r", errors="replace") as f:
                content = f.read()

            if old_string not in content:
                return ToolResult(f"oldString not found in {file_path}", error=True)

            count = content.count(old_string)
            if count > 1 and not replace_all:
                return ToolResult(
                    f"Found {count} matches in {file_path}. Use replaceAll=true or provide more context.",
                    error=True,
                )

            if replace_all:
                content = content.replace(old_string, new_string)
            else:
                content = content.replace(old_string, new_string, 1)

            with open(file_path, "w") as f:
                f.write(content)

            return ToolResult(f"Edited {file_path}")
        except Exception as e:
            return ToolResult(f"Error editing file: {e}", error=True)

    def _tool_write(self, args: dict) -> ToolResult:
        file_path = args.get("file_path", "")
        content = args.get("content", "")

        if not file_path:
            return ToolResult("No file path provided", error=True)

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)
            return ToolResult(f"Wrote {file_path}")
        except Exception as e:
            return ToolResult(f"Error writing file: {e}", error=True)

    def _tool_glob(self, args: dict) -> ToolResult:
        pattern = args.get("pattern", "**/*")
        path = args.get("path", self.workdir)

        try:
            matches = globmod.glob(os.path.join(path, pattern), recursive=True)
            matches = [m for m in matches if os.path.isfile(m)]
            matches.sort(key=os.path.getmtime, reverse=True)

            if len(matches) > 200:
                matches = matches[:200]

            result = "".join(f"  {m}" for m in matches) if matches else "No files found"
            return ToolResult(result)
        except Exception as e:
            return ToolResult(f"Error globbing: {e}", error=True)

    def _tool_grep(self, args: dict) -> ToolResult:
        pattern = args.get("pattern", "")
        path = args.get("path", self.workdir)
        include = args.get("include", "")

        if not pattern:
            return ToolResult("No pattern provided", error=True)

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(f"Invalid regex: {e}", error=True)

        matches = []
        try:
            for root, _dirs, files in os.walk(path):
                if ".git" in root.split(os.sep):
                    continue
                for fname in files:
                    if include and not fnmatch.fnmatch(fname, include):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, errors="replace") as f:
                            for i, line in enumerate(f, 1):
                                if regex.search(line):
                                    matches.append(f"{fpath}:{i}: {line.rstrip()}")
                                    if len(matches) >= 200:
                                        break
                    except (OSError, UnicodeDecodeError):
                        continue
                if len(matches) >= 200:
                    break
        except Exception as e:
            return ToolResult(f"Error searching: {e}", error=True)

        if not matches:
            return ToolResult("No matches found")
        return ToolResult("".join(f"  {m}" for m in matches))

    def _tool_apply_patch(self, args: dict) -> ToolResult:
        patch_text = args.get("patchText", "")
        if not patch_text:
            return ToolResult("No patch text provided", error=True)

        results = []
        current_file = None
        current_content = None
        new_file_content = []

        for line in patch_text.splitlines():
            if line.startswith("*** Add File: "):
                current_file = line[len("*** Add File: "):].strip()
                current_content = None
                new_file_content = []
            elif line.startswith("*** Update File: "):
                current_file = line[len("*** Update File: "):].strip()
                if os.path.isfile(current_file):
                    with open(current_file, errors="replace") as f:
                        current_content = f.read()
                new_file_content = []
            elif line.startswith("*** Delete File: "):
                fpath = line[len("*** Delete File: "):].strip()
                try:
                    os.unlink(fpath)
                    results.append(f"Deleted {fpath}")
                except Exception as e:
                    results.append(f"Error deleting {fpath}: {e}")
            elif line.startswith("*** Move to: "):
                new_path = line[len("*** Move to: "):].strip()
                if current_file and current_content is not None:
                    try:
                        os.makedirs(os.path.dirname(new_path), exist_ok=True)
                        with open(new_path, "w") as f:
                            f.write(current_content)
                        results.append(f"Moved {current_file} to {new_path}")
                    except Exception as e:
                        results.append(f"Error moving: {e}")
            elif line.startswith("+"):
                new_file_content.append(line[1:])
            elif line.startswith("-"):
                pass
            else:
                new_file_content.append(line)

        if current_file and new_file_content:
            try:
                os.makedirs(os.path.dirname(current_file), exist_ok=True)
                with open(current_file, "w") as f:
                    f.write("".join(new_file_content))
                results.append(f"Patched {current_file}")
            except Exception as e:
                results.append(f"Error patching {current_file}: {e}")

        if not results:
            return ToolResult("No actions in patch", error=True)
        return ToolResult("; ".join(results))

    def _tool_webfetch(self, args: dict) -> ToolResult:
        url = args.get("url", "")
        if not url:
            return ToolResult("No URL provided", error=True)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tq-chat/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode(errors="replace")

            if len(content) > 51200:
                content = content[:51200] + "...(truncated)"

            return ToolResult(content)
        except Exception as e:
            return ToolResult(f"Error fetching URL: {e}", error=True)

    def _tool_websearch(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        if not query:
            return ToolResult("No query provided", error=True)

        try:
            import httpx
        except ImportError:
            return ToolResult("httpx is required for websearch. Install with: pip install tq-serve[chat]", error=True)

        try:
            with httpx.Client(timeout=15) as client:
                resp = client.post(
                    "https://mcp.exa.ai/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "id": 1,
                        "params": {
                            "name": "web_search_exa",
                            "arguments": {"query": query, "type": "auto"},
                        },
                    },
                    headers={"Content-Type": "application/json"},
                )
                data = resp.json()

            results = data.get("result", {})
            content = results.get("content", [])
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    elif isinstance(item, str):
                        texts.append(item)
                output = "".join(texts)
            else:
                output = str(content)

            if not output:
                output = "No results found"

            if len(output) > 51200:
                output = output[:51200] + "...(truncated)"

            return ToolResult(output)
        except Exception as e:
            return ToolResult(f"Web search error: {e}", error=True)

    def _tool_question(self, args: dict) -> ToolResult:
        question_text = args.get("question", "")
        header = args.get("header", "Question")

        print(f"  [{header}] {question_text}")
        try:
            answer = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            answer = ""

        return ToolResult(answer)

    def _tool_todowrite(self, args: dict) -> ToolResult:
        todos = args.get("todos", [])
        if not todos:
            return ToolResult("No todos provided", error=True)

        lines = []
        for t in todos:
            content = t.get("content", "") if isinstance(t, dict) else str(t)
            status = t.get("status", "pending") if isinstance(t, dict) else "pending"
            lines.append(f"  [{status[0].upper()}] {content}")

        output = "".join(lines)
        print(output)
        return ToolResult(f"Updated todo list ({len(todos)} items)")
