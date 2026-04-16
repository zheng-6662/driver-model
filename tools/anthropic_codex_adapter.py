from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import requests


DEFAULT_BACKEND_BASE_URL = os.environ.get(
    "CLAUDE_OPENAI_BACKEND_BASE_URL", "http://localhost:8317/v1"
).rstrip("/")
DEFAULT_BACKEND_API_KEY = os.environ.get(
    "CLAUDE_OPENAI_BACKEND_API_KEY", "sk-dummy"
)
DEFAULT_BACKEND_MODEL = os.environ.get("CLAUDE_OPENAI_MODEL", "gpt-5.4")
DEFAULT_PROXY_API_KEY = os.environ.get(
    "CLAUDE_OPENAI_PROXY_API_KEY", "sk-claude-codex-proxy"
)
DEFAULT_REQUEST_TIMEOUT = float(
    os.environ.get("CLAUDE_OPENAI_BACKEND_TIMEOUT_SECONDS", "600")
)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def transport_json_dumps(payload: dict[str, Any]) -> str:
    # Keep transport payloads ASCII-only so Windows clients that mishandle
    # raw UTF-8 bytes in streamed JSON still reconstruct Unicode correctly.
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_anthropic_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = []
    for item in ensure_list(content):
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            blocks.append(item)
    return blocks


def anthropic_system_to_instructions(system_field: Any) -> str | None:
    if system_field is None:
        return None

    if isinstance(system_field, str):
        text = system_field.strip()
        return text or None

    texts: list[str] = []
    for block in ensure_list(system_field):
        if isinstance(block, str):
            if block.strip():
                texts.append(block.strip())
            continue
        if isinstance(block, dict) and block.get("type") == "text":
            block_text = str(block.get("text", "")).strip()
            if block_text:
                texts.append(block_text)
    if not texts:
        return None
    return "\n\n".join(texts)


def stringify_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        text_parts: list[str] = []
        other_parts: list[Any] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            else:
                other_parts.append(item)
        if other_parts and not text_parts:
            return json.dumps(content, ensure_ascii=False)
        if other_parts:
            return "\n".join(text_parts + [json.dumps(other_parts, ensure_ascii=False)])
        return "\n".join(part for part in text_parts if part)
    return str(content)


def default_filename_for_media_type(media_type: str | None, fallback: str) -> str:
    media_type = (media_type or "").lower()
    extension_map = {
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "text/csv": ".csv",
        "application/json": ".json",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    extension = extension_map.get(media_type, "")
    if fallback.lower().endswith(extension) or not extension:
        return fallback
    return fallback + extension


def anthropic_image_block_to_openai(block: dict[str, Any]) -> dict[str, Any]:
    source = block.get("source") or {}
    if source.get("type") == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return {"type": "input_image", "image_url": f"data:{media_type};base64,{data}"}
    if source.get("type") == "url":
        return {"type": "input_image", "image_url": source.get("url", "")}
    if source.get("type") in {"file", "file_id"} and source.get("file_id"):
        return {"type": "input_image", "file_id": source.get("file_id")}
    raise ValueError(f"Unsupported image source type: {source.get('type')!r}")


def anthropic_document_block_to_openai(block: dict[str, Any]) -> dict[str, Any]:
    source = block.get("source") or {}
    title = str(block.get("title") or "document")
    source_type = source.get("type")

    if source_type == "url":
        return {"type": "input_file", "file_url": source.get("url", "")}

    if source_type == "base64":
        media_type = source.get("media_type")
        filename = default_filename_for_media_type(media_type, title)
        return {
            "type": "input_file",
            "filename": filename,
            "file_data": source.get("data", ""),
        }

    if source_type in {"file", "file_id"} and source.get("file_id"):
        return {"type": "input_file", "file_id": source.get("file_id")}

    raise ValueError(f"Unsupported document source type: {source_type!r}")


def flush_message_item(
    items: list[dict[str, Any]], role: str, content_blocks: list[dict[str, Any]]
) -> None:
    if not content_blocks:
        return
    items.append({"role": role, "content": content_blocks[:]})
    content_blocks.clear()


def convert_user_message_to_openai_items(
    blocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    current_content: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            current_content.append(
                {"type": "input_text", "text": str(block.get("text", ""))}
            )
            continue

        if block_type == "image":
            current_content.append(anthropic_image_block_to_openai(block))
            continue

        if block_type == "document":
            current_content.append(anthropic_document_block_to_openai(block))
            continue

        if block_type == "tool_result":
            flush_message_item(items, "user", current_content)
            tool_use_id = str(block.get("tool_use_id", ""))
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_use_id,
                    "output": stringify_tool_result_content(block.get("content")),
                }
            )
            continue

        raise ValueError(f"Unsupported user block type: {block_type!r}")

    flush_message_item(items, "user", current_content)
    return items


def convert_assistant_message_to_openai_items(
    blocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    current_content: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            current_content.append(
                {"type": "output_text", "text": str(block.get("text", ""))}
            )
            continue

        if block_type == "thinking":
            thinking_text = str(block.get("thinking", "")).strip()
            if thinking_text:
                current_content.append({"type": "output_text", "text": thinking_text})
            continue

        if block_type == "redacted_thinking":
            continue

        if block_type == "tool_use":
            flush_message_item(items, "assistant", current_content)
            tool_input = block.get("input", {})
            if isinstance(tool_input, str):
                arguments = tool_input
            else:
                arguments = json.dumps(
                    tool_input, ensure_ascii=False, separators=(",", ":")
                )
            tool_use_id = str(block.get("id") or block.get("tool_use_id") or uuid.uuid4())
            items.append(
                {
                    "type": "function_call",
                    "call_id": tool_use_id,
                    "name": str(block.get("name", "")),
                    "arguments": arguments,
                }
            )
            continue

        raise ValueError(f"Unsupported assistant block type: {block_type!r}")

    flush_message_item(items, "assistant", current_content)
    return items


def anthropic_messages_to_openai_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        blocks = normalize_anthropic_blocks(message.get("content"))
        if role == "user":
            items.extend(convert_user_message_to_openai_items(blocks))
        elif role == "assistant":
            items.extend(convert_assistant_message_to_openai_items(blocks))
        else:
            raise ValueError(f"Unsupported message role: {role!r}")
    return items


def anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in ensure_list(tools):
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        converted.append(
            {
                "type": "function",
                "name": str(name),
                "description": str(tool.get("description", "")),
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            }
        )
    return converted


def anthropic_tool_choice_to_openai(choice: Any) -> Any:
    if choice is None:
        return None
    if isinstance(choice, str):
        return choice
    if not isinstance(choice, dict):
        return None

    choice_type = choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool" and choice.get("name"):
        return {"type": "function", "name": choice["name"]}
    return None


def estimate_input_tokens(body: dict[str, Any]) -> int:
    snippets: list[str] = []
    instructions = anthropic_system_to_instructions(body.get("system"))
    if instructions:
        snippets.append(instructions)

    for message in ensure_list(body.get("messages")):
        for block in normalize_anthropic_blocks(message.get("content")):
            if block.get("type") == "text":
                snippets.append(str(block.get("text", "")))
            elif block.get("type") == "tool_use":
                snippets.append(json.dumps(block.get("input", {}), ensure_ascii=False))
            elif block.get("type") == "tool_result":
                snippets.append(stringify_tool_result_content(block.get("content")))

    if body.get("tools"):
        snippets.append(json.dumps(body.get("tools"), ensure_ascii=False))

    total_chars = sum(len(item) for item in snippets)
    return max(1, math.ceil(total_chars / 4))


class BackendStreamAccumulator:
    def __init__(self, request_model: str) -> None:
        self.request_model = request_model
        self.backend_response_id = f"msg_proxy_{uuid.uuid4().hex}"
        self.content_blocks: list[dict[str, Any]] = []
        self.item_to_index: dict[str, int] = {}
        self.tool_arg_buffers: dict[str, str] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason = "end_turn"

    def start_text_block(self, item_id: str) -> int:
        index = len(self.content_blocks)
        self.content_blocks.append({"type": "text", "text": ""})
        self.item_to_index[item_id] = index
        return index

    def start_tool_block(self, item: dict[str, Any]) -> int:
        item_id = str(item.get("id", ""))
        call_id = str(item.get("call_id") or item_id or uuid.uuid4())
        index = len(self.content_blocks)
        self.content_blocks.append(
            {
                "type": "tool_use",
                "id": call_id,
                "name": str(item.get("name", "")),
                "input": {},
            }
        )
        self.item_to_index[item_id] = index
        self.tool_arg_buffers[item_id] = ""
        self.stop_reason = "tool_use"
        return index

    def append_text_delta(self, item_id: str, delta: str) -> int:
        index = self.item_to_index[item_id]
        self.content_blocks[index]["text"] += delta
        return index

    def append_tool_delta(self, item_id: str, delta: str) -> int:
        self.tool_arg_buffers[item_id] = self.tool_arg_buffers.get(item_id, "") + delta
        return self.item_to_index[item_id]

    def complete_tool_block(self, item: dict[str, Any]) -> int:
        item_id = str(item.get("id", ""))
        index = self.item_to_index[item_id]
        raw_arguments = self.tool_arg_buffers.get(item_id) or str(item.get("arguments", ""))
        try:
            parsed = json.loads(raw_arguments) if raw_arguments else {}
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
        except json.JSONDecodeError:
            parsed = {"_raw": raw_arguments}
        self.content_blocks[index]["input"] = parsed
        return index

    def set_usage(self, usage: dict[str, Any] | None) -> None:
        usage = usage or {}
        self.input_tokens = int(usage.get("input_tokens", 0) or 0)
        self.output_tokens = int(usage.get("output_tokens", 0) or 0)

    def build_message(self) -> dict[str, Any]:
        return {
            "id": self.backend_response_id,
            "type": "message",
            "role": "assistant",
            "model": self.request_model,
            "content": self.content_blocks,
            "stop_reason": self.stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
        }


def iter_sse_events(response: requests.Response):
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                payload_text = "\n".join(data_lines)
                yield event_name, json.loads(payload_text)
            event_name = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())

    if data_lines:
        payload_text = "\n".join(data_lines)
        yield event_name, json.loads(payload_text)


class AnthropicCodexProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AnthropicCodexAdapter/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        log(
            "%s - - [%s] %s"
            % (self.client_address[0], self.log_date_time_string(), fmt % args)
        )

    def do_GET(self) -> None:
        route = urlparse(self.path).path

        if route in {"", "/"}:
            self.send_json(200, {"ok": True, "service": "anthropic-codex-adapter"})
            return

        if route == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "backend_base_url": self.server.backend_base_url,
                    "backend_model": self.server.backend_model,
                },
            )
            return

        if route == "/v1/models":
            self.proxy_models()
            return

        self.send_json(404, {"error": {"type": "not_found", "message": "Not found"}})

    def do_HEAD(self) -> None:
        route = urlparse(self.path).path
        if route in {"", "/", "/health"}:
            self.close_connection = True
            self.send_response(200)
            self.send_header("Connection", "close")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.close_connection = True
        self.send_response(404)
        self.send_header("Connection", "close")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        route = urlparse(self.path).path

        if not self.is_authorized():
            self.send_json(
                401,
                {
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "Invalid proxy API key"},
                },
            )
            return

        if route == "/v1/messages":
            self.handle_messages()
            return

        if route == "/v1/messages/count_tokens":
            self.handle_count_tokens()
            return

        self.send_json(404, {"error": {"type": "not_found", "message": "Not found"}})

    def is_authorized(self) -> bool:
        expected = self.server.proxy_api_key
        if not expected:
            return True

        x_api_key = self.headers.get("x-api-key")
        if x_api_key == expected:
            return True

        auth_header = self.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            bearer = auth_header.split(" ", 1)[1].strip()
            if bearer == expected:
                return True

        return False

    def read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))

    def send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = transport_json_dumps(payload).encode("utf-8")
        self.close_connection = True
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def send_sse_event(self, event_name: str, payload: dict[str, Any]) -> None:
        body = (
            f"event: {event_name}\n"
            f"data: {transport_json_dumps(payload)}\n\n"
        ).encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def handle_count_tokens(self) -> None:
        try:
            body = self.read_json_body()
            count = estimate_input_tokens(body)
            self.send_json(200, {"input_tokens": count})
        except Exception as exc:
            self.send_json(
                400,
                {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}},
            )

    def handle_messages(self) -> None:
        try:
            anthropic_body = self.read_json_body()
            backend_payload = self.build_backend_payload(anthropic_body)
            request_model = str(anthropic_body.get("model") or self.server.backend_model)
            accumulator = BackendStreamAccumulator(request_model=request_model)
            stream_requested = bool(anthropic_body.get("stream"))

            if stream_requested:
                self.close_connection = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self.send_sse_event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": accumulator.backend_response_id,
                            "type": "message",
                            "role": "assistant",
                            "model": request_model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                )

            self.stream_backend_to_accumulator(
                backend_payload=backend_payload,
                accumulator=accumulator,
                stream_to_client=stream_requested,
            )

            if stream_requested:
                self.send_sse_event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": accumulator.stop_reason,
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": accumulator.output_tokens},
                    },
                )
                self.send_sse_event("message_stop", {"type": "message_stop"})
                return

            self.send_json(200, accumulator.build_message())
        except requests.HTTPError as exc:
            self.handle_backend_http_error(exc)
        except Exception as exc:
            log(traceback.format_exc())
            self.send_json(
                500,
                {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
            )

    def proxy_models(self) -> None:
        try:
            response = requests.get(
                f"{self.server.backend_base_url}/models",
                headers={"Authorization": f"Bearer {self.server.backend_api_key}"},
                timeout=self.server.request_timeout,
            )
            response.raise_for_status()
            raw = response.content
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()
        except requests.HTTPError as exc:
            self.handle_backend_http_error(exc)
        except Exception as exc:
            self.send_json(
                500,
                {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
            )

    def build_backend_payload(self, anthropic_body: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.server.backend_model,
            "input": anthropic_messages_to_openai_input(
                ensure_list(anthropic_body.get("messages"))
            ),
            "stream": True,
        }

        instructions = anthropic_system_to_instructions(anthropic_body.get("system"))
        if instructions:
            payload["instructions"] = instructions

        tools = anthropic_tools_to_openai(anthropic_body.get("tools"))
        if tools:
            payload["tools"] = tools

        tool_choice = anthropic_tool_choice_to_openai(anthropic_body.get("tool_choice"))
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if anthropic_body.get("max_tokens") is not None:
            payload["max_output_tokens"] = anthropic_body.get("max_tokens")

        if anthropic_body.get("temperature") is not None:
            payload["temperature"] = anthropic_body.get("temperature")

        if anthropic_body.get("top_p") is not None:
            payload["top_p"] = anthropic_body.get("top_p")

        return payload

    def stream_backend_to_accumulator(
        self,
        backend_payload: dict[str, Any],
        accumulator: BackendStreamAccumulator,
        stream_to_client: bool,
    ) -> None:
        headers = {
            "Authorization": f"Bearer {self.server.backend_api_key}",
            "Accept": "text/event-stream",
        }
        if self.server.verbose:
            log(f"Proxy request -> {self.server.backend_base_url}/responses")

        with requests.post(
            f"{self.server.backend_base_url}/responses",
            headers=headers,
            json=backend_payload,
            stream=True,
            timeout=self.server.request_timeout,
        ) as response:
            response.raise_for_status()
            for _, event_payload in iter_sse_events(response):
                event_type = event_payload.get("type")

                if self.server.verbose:
                    log(f"Backend event: {event_type}")

                if event_type == "response.output_item.added":
                    item = event_payload["item"]
                    item_type = item.get("type")
                    if item_type == "message":
                        index = accumulator.start_text_block(item["id"])
                        if stream_to_client:
                            self.send_sse_event(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": index,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                    elif item_type == "function_call":
                        index = accumulator.start_tool_block(item)
                        if stream_to_client:
                            self.send_sse_event(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": accumulator.content_blocks[index]["id"],
                                        "name": accumulator.content_blocks[index]["name"],
                                        "input": {},
                                    },
                                },
                            )
                    continue

                if event_type == "response.output_text.delta":
                    index = accumulator.append_text_delta(
                        event_payload["item_id"], event_payload.get("delta", "")
                    )
                    if stream_to_client:
                        self.send_sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "text_delta",
                                    "text": event_payload.get("delta", ""),
                                },
                            },
                        )
                    continue

                if event_type == "response.function_call_arguments.delta":
                    index = accumulator.append_tool_delta(
                        event_payload["item_id"], event_payload.get("delta", "")
                    )
                    if stream_to_client:
                        self.send_sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": event_payload.get("delta", ""),
                                },
                            },
                        )
                    continue

                if event_type == "response.output_item.done":
                    item = event_payload["item"]
                    item_type = item.get("type")
                    if item_type == "function_call":
                        index = accumulator.complete_tool_block(item)
                    else:
                        index = accumulator.item_to_index.get(item.get("id"), -1)
                    if stream_to_client and index >= 0:
                        self.send_sse_event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": index},
                        )
                    continue

                if event_type == "response.completed":
                    accumulator.set_usage(event_payload.get("response", {}).get("usage"))
                    continue

    def handle_backend_http_error(self, exc: requests.HTTPError) -> None:
        response = exc.response
        status = response.status_code if response is not None else 502
        detail = ""
        if response is not None:
            detail = response.text
        self.send_json(
            status,
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": detail or str(exc),
                },
            },
        )


class AnthropicCodexProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        backend_base_url: str,
        backend_api_key: str,
        backend_model: str,
        proxy_api_key: str,
        request_timeout: float,
        verbose: bool,
    ) -> None:
        super().__init__(server_address, AnthropicCodexProxyHandler)
        self.backend_base_url = backend_base_url.rstrip("/")
        self.backend_api_key = backend_api_key
        self.backend_model = backend_model
        self.proxy_api_key = proxy_api_key
        self.request_timeout = request_timeout
        self.verbose = verbose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anthropic Messages API adapter backed by an OpenAI/Codex Responses endpoint."
    )
    parser.add_argument("--host", default=os.environ.get("CLAUDE_OPENAI_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CLAUDE_OPENAI_PORT", "8417")),
    )
    parser.add_argument("--backend-base-url", default=DEFAULT_BACKEND_BASE_URL)
    parser.add_argument("--backend-api-key", default=DEFAULT_BACKEND_API_KEY)
    parser.add_argument("--backend-model", default=DEFAULT_BACKEND_MODEL)
    parser.add_argument("--proxy-api-key", default=DEFAULT_PROXY_API_KEY)
    parser.add_argument("--timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--verbose", action="store_true", default=env_flag("CLAUDE_OPENAI_VERBOSE"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = AnthropicCodexProxyServer(
        server_address=(args.host, args.port),
        backend_base_url=args.backend_base_url,
        backend_api_key=args.backend_api_key,
        backend_model=args.backend_model,
        proxy_api_key=args.proxy_api_key,
        request_timeout=args.timeout,
        verbose=args.verbose,
    )
    log(
        "Anthropic Codex adapter listening on "
        f"http://{args.host}:{args.port} -> {args.backend_base_url} "
        f"(model={args.backend_model})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down adapter.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
