from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Generator, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def parsed_args(self) -> dict:
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError:
            return {}


@dataclass
class ChatMessage:
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class StreamChunk:
    delta_content: str = ""
    tool_call_id: Optional[str] = None
    tool_call_name: str = ""
    tool_call_args: str = ""
    finish_reason: Optional[str] = None


class ChatClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get_httpx(self):
        try:
            import httpx
            return httpx
        except ImportError:
            raise ImportError(
                "httpx is required for tq chat. Install with: pip install tq-serve[chat]"
            )

    def is_available(self) -> bool:
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> list[dict]:
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            return data.get("data", [])
        except Exception:
            return []

    def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str = "",
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Generator[StreamChunk, None, None]:
        httpx = self._get_httpx()

        payload = {
            "messages": [self._message_to_dict(m) for m in messages],
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            payload["model"] = model
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                chunk = StreamChunk(finish_reason=finish_reason)

                if "content" in delta and delta["content"] is not None:
                    chunk.delta_content = delta["content"]

                if "tool_calls" in delta and delta["tool_calls"]:
                    tc = delta["tool_calls"][0]
                    chunk.tool_call_id = tc.get("id")
                    if "function" in tc:
                        fname = tc["function"].get("name", "")
                        fargs = tc["function"].get("arguments", "")
                        if fname:
                            chunk.tool_call_name = fname
                        if fargs:
                            chunk.tool_call_args = fargs

                yield chunk

    def _message_to_dict(self, msg: ChatMessage) -> dict:
        d: dict = {"role": msg.role}
        if msg.content:
            d["content"] = msg.content
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.name:
            d["name"] = msg.name
        return d
