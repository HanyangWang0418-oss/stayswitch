"""Anthropic-format backend that returns a recorded response.

LiteLLM's ``mock_response`` on /v1/messages is text-only and never streams, so a replayed tool call
would reach Claude Code as an empty turn. The proxy instead routes replayed calls to this server, which
rebuilds the recorded response (text plus tool calls) as a real Anthropic message, streamed or not.
The server is stateless: the recording travels in the ``x-stayswitch-replay`` header, so any number of
proxies can share one instance.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

REPLAY_HEADER = "x-stayswitch-replay"
DEFAULT_PORT = 4390


def encode_recording(text: str, tool_calls: list[dict[str, Any]] | None, extra: dict[str, Any] | None = None) -> str:
    payload = json.dumps({"text": text or "", "tool_calls": tool_calls or [], "extra": extra or {}})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def anthropic_message(recording: dict[str, Any], model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    extra = recording.get("extra") or {}
    if extra.get("thinking"):
        content.append({"type": "thinking", "thinking": extra["thinking"], "signature": ""})
    if recording["text"]:
        content.append({"type": "text", "text": recording["text"]})
    for call in recording["tool_calls"]:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content.append({"type": "tool_use", "id": call["id"], "name": fn.get("name", ""), "input": args})
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "id": f"msg_replay_{time.time_ns()}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use" if recording["tool_calls"] else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": extra.get("input_tokens", 0),
            "output_tokens": extra.get("output_tokens", 0),
            "cache_read_input_tokens": extra.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": 0,
        },
    }


def sse_events(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}})
    ]
    for i, block in enumerate(message["content"]):
        if block["type"] == "text":
            start, delta = {"type": "text", "text": ""}, {"type": "text_delta", "text": block["text"]}
        elif block["type"] == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        else:
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        events += [
            ("content_block_start", {"type": "content_block_start", "index": i, "content_block": start}),
            ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": i}),
        ]
    events += [
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None}, "usage": {"output_tokens": message["usage"]["output_tokens"]}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return events


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        raw = self.headers.get(REPLAY_HEADER)
        if not raw:
            self.send_error(400, "missing replay header")
            return
        message = anthropic_message(json.loads(base64.urlsafe_b64decode(raw)), body.get("model", "replay"))
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for name, data in sse_events(message):
                self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            return
        data = json.dumps(message).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


_lock = threading.Lock()
_started = False


def ensure_started(port: int = DEFAULT_PORT) -> None:
    """Start the server in this process unless one already listens (another proxy may own the port)."""
    global _started
    with _lock:
        if _started:
            return
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        except OSError:
            _started = True
            return
        threading.Thread(target=server.serve_forever, daemon=True, name="stayswitch-replay").start()
        _started = True
