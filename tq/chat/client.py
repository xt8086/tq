from __future__ import annotations

import base64
import json
import os
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
    images: list[str] = field(default_factory=list)


def load_image_as_base64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime = mime_map.get(ext, "image/png")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return None


def load_pdf_as_images(path: str) -> list[str]:
    try:
        import fitz
        doc = fitz.open(path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode()
            images.append(f"data:image/png;base64,{b64}")
            if len(images) >= 10:
                break
        doc.close()
        if images:
            return images
    except ImportError:
        pass
    except Exception:
        pass
    try:
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["pdftoppm", "-png", "-r", "150", path, os.path.join(tmpdir, "page")],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                images = []
                for f in sorted(os.listdir(tmpdir)):
                    if f.endswith(".png"):
                        img = load_image_as_base64(os.path.join(tmpdir, f))
                        if img:
                            images.append(img)
                if images:
                    return images
    except Exception:
        pass
    return []


def load_docx_as_text(path: str) -> Optional[str]:
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(path) as z:
            with z.open("word/document.xml") as f:
                tree = ET.parse(f)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for p in tree.findall(".//w:p", ns):
            texts = [t.text for t in p.findall(".//w:t", ns) if t.text]
            if texts:
                paragraphs.append("".join(texts))
        return "\n".join(paragraphs) if paragraphs else None
    except Exception:
        return None


def load_xlsx_as_text(path: str) -> Optional[str]:
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        rows = []
        with zipfile.ZipFile(path) as z:
            shared = {}
            if "xl/sharedStrings.xml" in z.namelist():
                with z.open("xl/sharedStrings.xml") as f:
                    stree = ET.parse(f)
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for i, si in enumerate(stree.findall(".//s:si", ns)):
                    texts = [t.text for t in si.findall(".//s:t", ns) if t.text]
                    shared[i] = "".join(texts)
            sheet_paths = [n for n in sorted(z.namelist()) if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
            for sp in sheet_paths:
                with z.open(sp) as f:
                    sheet = ET.parse(f)
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for row in sheet.findall(".//s:row", ns):
                    cells = []
                    for cell in row.findall("s:c", ns):
                        v = cell.find("s:v", ns)
                        t_attr = cell.get("t", "")
                        val = v.text if v is not None else ""
                        if t_attr == "s" and val.isdigit():
                            val = shared.get(int(val), val)
                        cells.append(val)
                    rows.append("\t".join(cells))
                rows.append("")
        return "\n".join(rows) if rows else None
    except Exception:
        return None


def load_csv_as_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(50000)
    except Exception:
        return None


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
        if msg.images:
            content_parts = []
            if msg.content:
                content_parts.append({"type": "text", "text": msg.content})
            for img_url in msg.images:
                content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
            d["content"] = content_parts
        elif msg.content:
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
