"""OpenAI Responses API provider (streaming) for backends like codex-for.me."""

import json
from typing import Any

import openai
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class ResponsesProvider(LLMProvider):
    """
    Provider that calls the OpenAI *Responses* API with streaming.

    Uses the official ``openai`` SDK with ``stream=True`` — the same approach
    as Codex CLI (``wire_api = "responses"``).

    Reference: https://docs.codex-for.me/openclaw/
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-5.2",
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._client = openai.AsyncOpenAI(
            api_key=api_key or "",
            base_url=api_base,
            default_headers=extra_headers or {},
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        instructions, input_items = self._messages_to_input(messages)
        resp_tools = self._convert_tools(tools)

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "input": input_items,
            "stream": True,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if resp_tools:
            kwargs["tools"] = resp_tools

        logger.debug(f"[ResponsesProvider] model={kwargs['model']}  inputs={len(input_items)}")

        try:
            stream = await self._client.responses.create(**kwargs)
            return await self._consume_stream(stream)
        except Exception as e:
            logger.error(f"[ResponsesProvider] {type(e).__name__}: {e}")
            return LLMResponse(content=f"Error calling Responses API: {e}", finish_reason="error")

    def get_default_model(self) -> str:
        return self.default_model

    # ------------------------------------------------------------------
    # Stream consumption
    # ------------------------------------------------------------------

    async def _consume_stream(self, stream: Any) -> LLMResponse:
        """Iterate over SDK streaming events and aggregate into LLMResponse."""
        content_parts: list[str] = []
        # Internal map key: function_call item id (prefer `fc_*`), with external call id (`call_*`) kept separately.
        tool_calls: dict[str, dict[str, Any]] = {}
        usage: dict[str, int] = {}

        async for event in stream:
            etype = getattr(event, "type", "")
            logger.debug(f"[ResponsesProvider] event: {etype}")

            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                logger.debug(f"[ResponsesProvider] text delta: {delta!r}")
                content_parts.append(delta)

            elif etype == "response.output_item.added":
                item = event.item
                item_type = getattr(item, "type", "")
                item_id = getattr(item, "id", "") or getattr(item, "call_id", "")
                external_call_id = getattr(item, "call_id", "") or item_id
                logger.debug(
                    f"[ResponsesProvider] output_item.added: type={item_type!r}  "
                    f"item_id={item_id!r}  call_id={external_call_id!r}  name={getattr(item, 'name', '')!r}  "
                    f"item_attrs={[a for a in dir(item) if not a.startswith('_')]}"
                )
                if item_type == "function_call":
                    tool_calls[item_id] = {
                        "name": getattr(item, "name", "") or "",
                        "external_call_id": external_call_id,
                        "arguments_raw": "",
                    }

            elif etype == "response.function_call_arguments.delta":
                # In codex-for.me, delta event call_id often uses `fc_*` item id, not `call_*`.
                call_id = getattr(event, "item_id", "") or getattr(event, "call_id", "")
                logger.debug(
                    f"[ResponsesProvider] fn_args.delta: call_id={call_id!r}  "
                    f"known_ids={list(tool_calls.keys())}  delta={getattr(event, 'delta', '')!r}"
                )
                if call_id in tool_calls:
                    tool_calls[call_id]["arguments_raw"] += getattr(event, "delta", "")
                else:
                    tool_calls.setdefault(call_id, {"name": "", "external_call_id": call_id, "arguments_raw": ""})
                    tool_calls[call_id]["arguments_raw"] += getattr(event, "delta", "")

            elif etype == "response.function_call_arguments.done":
                call_id = getattr(event, "item_id", "") or getattr(event, "call_id", "")
                logger.debug(
                    f"[ResponsesProvider] fn_args.done: call_id={call_id!r}  "
                    f"known_ids={list(tool_calls.keys())}  "
                    f"name_attr={getattr(event, 'name', 'N/A')!r}  "
                    f"event_attrs={[a for a in dir(event) if not a.startswith('_')]}"
                )
                if call_id in tool_calls:
                    raw = tool_calls[call_id].pop("arguments_raw", "{}")
                    try:
                        tool_calls[call_id]["arguments"] = json.loads(raw)
                    except json.JSONDecodeError:
                        tool_calls[call_id]["arguments"] = {"raw": raw}
                    event_name = getattr(event, "name", None)
                    if not tool_calls[call_id].get("name") and event_name:
                        tool_calls[call_id]["name"] = event_name

            elif etype == "response.completed":
                resp = event.response
                u = getattr(resp, "usage", None)
                if u:
                    usage = {
                        "prompt_tokens": getattr(u, "input_tokens", 0),
                        "completion_tokens": getattr(u, "output_tokens", 0),
                        "total_tokens": getattr(u, "total_tokens", 0),
                    }
                # Fallback: extract tool calls from completed response output
                for output_item in getattr(resp, "output", []) or []:
                    if getattr(output_item, "type", "") == "function_call":
                        item_id = getattr(output_item, "id", "") or getattr(output_item, "call_id", "")
                        external_call_id = getattr(output_item, "call_id", "") or item_id
                        fc_id = item_id
                        if fc_id and fc_id not in tool_calls:
                            raw_args = getattr(output_item, "arguments", "{}")
                            try:
                                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                            except json.JSONDecodeError:
                                args = {"raw": raw_args}
                            tool_calls[fc_id] = {
                                "name": getattr(output_item, "name", "") or "",
                                "external_call_id": external_call_id,
                                "arguments": args,
                            }
                            logger.debug(f"[ResponsesProvider] fallback tool_call from completed: {fc_id}")

        # Finalize any tool_calls that still have unparsed arguments_raw
        for cid, info in tool_calls.items():
            if "arguments" not in info:
                raw = info.pop("arguments_raw", "{}")
                try:
                    info["arguments"] = json.loads(raw)
                except json.JSONDecodeError:
                    info["arguments"] = {"raw": raw} if raw else {}

        final_content = "".join(content_parts) or None
        final_tool_calls = [
            ToolCallRequest(
                id=info.get("external_call_id") or cid,
                name=info.get("name") or "",
                arguments=info.get("arguments", {}),
            )
            for cid, info in tool_calls.items()
            if (info.get("name") or "").strip()
        ]

        logger.debug(
            f"[ResponsesProvider] done  content_len={len(final_content or '')}  "
            f"tool_calls={len(final_tool_calls)}  usage={usage}"
        )

        return LLMResponse(
            content=final_content,
            tool_calls=final_tool_calls,
            finish_reason="tool_calls" if final_tool_calls else "stop",
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Chat Completions tool schema -> Responses function schema."""
        if not tools:
            return None
        converted: list[dict[str, Any]] = []
        for t in tools:
            fn = t.get("function", {})
            converted.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                "strict": False,
            })
        return converted

    @staticmethod
    def _messages_to_input(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert chat-style messages to Responses input items + instructions."""
        instructions: str | None = None
        items: list[dict[str, Any]] = []

        for m in messages:
            role = m.get("role")

            if role == "system":
                instructions = m.get("content")
                continue

            if role == "user":
                items.append({"role": "user", "content": m.get("content", "")})
                continue

            if role == "assistant":
                tool_calls = m.get("tool_calls") or []
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        fn_name = fn.get("name")
                        if not fn_name:
                            # Skip malformed historical tool call entries (e.g. name=None) to avoid 400.
                            continue
                        items.append({
                            "type": "function_call",
                            "call_id": tc.get("id"),
                            "name": fn_name,
                            "arguments": fn.get("arguments", "{}"),
                        })
                    if m.get("content"):
                        items.append({"role": "assistant", "content": m["content"]})
                else:
                    items.append({"role": "assistant", "content": m.get("content", "")})
                continue

            if role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id"),
                    "output": m.get("content", ""),
                })

        return instructions, items
