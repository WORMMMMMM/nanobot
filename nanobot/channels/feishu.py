"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import json
import re
import threading
from collections import OrderedDict
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        Emoji,
        GetMessageRequest,
        P2ImMessageReceiveV1,
        ReplyMessageRequest,
        ReplyMessageRequestBody,
    )
    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None
    Emoji = None

# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark channel using WebSocket long connection.
    
    Uses WebSocket to receive events - no public IP or webhook required.
    
    Requires:
    - App ID and App Secret from Feishu Open Platform
    - Bot capability enabled
    - Event subscription enabled (im.message.receive_v1)
    """
    
    name = "feishu"
    
    def __init__(self, config: FeishuConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()  # Ordered dedup cache
        self._loop: asyncio.AbstractEventLoop | None = None
    
    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return
        
        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return
        
        self._running = True
        self._loop = asyncio.get_running_loop()
        
        # Create Lark client for sending messages
        self._client = lark.Client.builder() \
            .app_id(self.config.app_id) \
            .app_secret(self.config.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        
        # Create event handler (only register message receive, ignore other events)
        event_handler = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(
            self._on_message_sync
        ).build()
        
        # Create WebSocket client for long connection
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO
        )
        
        # Start WebSocket client in a separate thread with reconnect loop
        def run_ws():
            while self._running:
                try:
                    self._ws_client.start()
                except Exception as e:
                    logger.warning(f"Feishu WebSocket error: {e}")
                if self._running:
                    import time; time.sleep(5)
        
        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        
        logger.info("Feishu bot started with WebSocket long connection")
        logger.info("No public IP required - using WebSocket to receive events")
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)
    
    async def stop(self) -> None:
        """Stop the Feishu bot."""
        self._running = False
        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception as e:
                logger.warning(f"Error stopping WebSocket client: {e}")
        logger.info("Feishu bot stopped")
    
    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> None:
        """Sync helper for adding reaction (runs in thread pool)."""
        try:
            request = CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                ).build()
            
            response = self._client.im.v1.message_reaction.create(request)
            
            if not response.success():
                logger.warning(f"Failed to add reaction: code={response.code}, msg={response.msg}")
            else:
                logger.debug(f"Added {emoji_type} reaction to message {message_id}")
        except Exception as e:
            logger.warning(f"Error adding reaction: {e}")

    async def _add_reaction(self, message_id: str, emoji_type: str = "THUMBSUP") -> None:
        """
        Add a reaction emoji to a message (non-blocking).
        
        Common emoji types: THUMBSUP, OK, EYES, DONE, OnIt, HEART
        """
        if not self._client or not Emoji:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._add_reaction_sync, message_id, emoji_type)

    @staticmethod
    def _decode_message_content(msg_type: str, raw_content: str | None) -> str:
        """Decode Feishu message body to plain text for LLM context."""
        if msg_type == "text":
            try:
                return json.loads(raw_content or "{}").get("text", "").strip()
            except json.JSONDecodeError:
                return (raw_content or "").strip()
        return MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")

    def _get_message_content_sync(self, message_id: str) -> tuple[str, str] | None:
        """Fetch a message by ID and return decoded content + type."""
        try:
            request = GetMessageRequest.builder().message_id(message_id).build()
            response = self._client.im.v1.message.get(request)
            if not response.success():
                logger.warning(
                    "Failed to fetch replied message: id={}, code={}, msg={}",
                    message_id,
                    response.code,
                    response.msg,
                )
                return None
            items = getattr(getattr(response, "data", None), "items", None) or []
            if not items:
                return None
            item = items[0]
            msg_type = getattr(item, "msg_type", "") or ""
            body = getattr(item, "body", None)
            raw_content = getattr(body, "content", "") if body else ""
            decoded = self._decode_message_content(msg_type, raw_content)
            return decoded, msg_type
        except Exception as e:
            logger.warning(f"Failed to fetch replied message {message_id}: {e}")
            return None

    async def _get_message_content(self, message_id: str) -> tuple[str, str] | None:
        """Async wrapper to fetch message details without blocking event loop."""
        if not self._client or not message_id:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_message_content_sync, message_id)
    
    # Regex to match markdown tables (header + separator + data rows)
    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown table into a Feishu table element."""
        lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            return None
        split = lambda l: [c.strip() for c in l.strip("|").split("|")]
        headers = split(lines[0])
        rows = [split(l) for l in lines[2:]]
        columns = [{"tag": "column", "name": f"c{i}", "display_name": h, "width": "auto"}
                   for i, h in enumerate(headers)]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [{f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))} for r in rows],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Split content into markdown + table elements for Feishu card."""
        elements, last_end = [], 0
        for m in self._TABLE_RE.finditer(content):
            before = content[last_end:m.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            elements.append(self._parse_md_table(m.group(1)) or {"tag": "markdown", "content": m.group(1)})
            last_end = m.end()
        remaining = content[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})
        return elements or [{"tag": "markdown", "content": content}]

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return
        
        try:
            # Build card with markdown + table support
            elements = self._build_card_elements(msg.content)
            card = {
                "config": {"wide_screen_mode": True},
                "elements": elements,
            }
            content = json.dumps(card, ensure_ascii=False)

            metadata = msg.metadata if isinstance(msg.metadata, dict) else {}
            feishu_meta = metadata.get("feishu") if isinstance(metadata.get("feishu"), dict) else {}
            reply_to_message_id = (
                msg.reply_to
                or feishu_meta.get("reply_to_message_id")
                or metadata.get("reply_to_message_id")
                or metadata.get("message_id")
            )

            if reply_to_message_id:
                request = ReplyMessageRequest.builder() \
                    .message_id(reply_to_message_id) \
                    .request_body(
                        ReplyMessageRequestBody.builder()
                        .msg_type("interactive")
                        .content(content)
                        .reply_in_thread(True)
                        .build()
                    ).build()
                response = self._client.im.v1.message.reply(request)
            else:
                # Determine receive_id_type based on chat_id format
                # open_id starts with "ou_", chat_id starts with "oc_"
                receive_id_type = "chat_id" if msg.chat_id.startswith("oc_") else "open_id"
                request = CreateMessageRequest.builder() \
                    .receive_id_type(receive_id_type) \
                    .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(msg.chat_id)
                        .msg_type("interactive")
                        .content(content)
                        .build()
                    ).build()
                response = self._client.im.v1.message.create(request)
            
            if not response.success():
                logger.error(
                    f"Failed to send Feishu message: code={response.code}, "
                    f"msg={response.msg}, log_id={response.get_log_id()}"
                )
            else:
                if reply_to_message_id:
                    logger.info(f"Feishu reply sent to message {reply_to_message_id}")
                else:
                    logger.info(f"Feishu message sent to {msg.chat_id}")
                
        except Exception as e:
            logger.error(f"Error sending Feishu message: {e}")
    
    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
        """
        Sync handler for incoming messages (called from WebSocket thread).
        Schedules async handling in the main event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)
    
    async def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        """Handle incoming message from Feishu."""
        try:
            event = data.event
            message = event.message
            sender = event.sender
            
            # Deduplication check
            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            
            # Trim cache: keep most recent 500 when exceeds 1000
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)
            
            # Skip bot messages
            sender_type = sender.sender_type
            if sender_type == "bot":
                return
            
            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type  # "p2p" or "group"
            msg_type = message.message_type
            parent_id = message.parent_id or ""
            root_id = message.root_id or ""
            thread_id = message.thread_id or ""
            
            # Add reaction to indicate "seen"
            await self._add_reaction(message_id, "THUMBSUP")
            
            # Parse message content
            content = self._decode_message_content(msg_type, message.content)
            
            if not content:
                return

            replied_message = await self._get_message_content(parent_id) if parent_id else None
            replied_text = replied_message[0] if replied_message else ""
            replied_msg_type = replied_message[1] if replied_message else ""
            if replied_text:
                content = f"[Replied Message]\n{replied_text}\n\n[New Message]\n{content}"

            reply_target = chat_id if chat_type == "group" else sender_id
            main_session_key = f"{self.name}:{reply_target}"
            thread_anchor = root_id or parent_id
            session_id = f"{main_session_key}:t:{thread_anchor}" if thread_anchor else None
            try:
                inherit_rounds = max(0, int(getattr(self.config, "thread_inherit_rounds", 6)))
            except (TypeError, ValueError):
                inherit_rounds = 6
            inherit_messages = inherit_rounds * 2

            logger.info(
                f"Feishu inbound message: sender={sender_id}, chat={chat_id}, "
                f"chat_type={chat_type}, msg_type={msg_type}, session={session_id or main_session_key}, "
                f"content={content[:120]!r}"
            )
            
            # Forward to message bus
            metadata = {
                "message_id": message_id,
                "reply_to_message_id": message_id,
                "chat_type": chat_type,
                "msg_type": msg_type,
                "parent_id": parent_id,
                "root_id": root_id,
                "thread_id": thread_id,
                "thread_anchor": thread_anchor,
                "feishu": {
                    "message_id": message_id,
                    "reply_to_message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                    "parent_id": parent_id,
                    "root_id": root_id,
                    "thread_id": thread_id,
                    "thread_anchor": thread_anchor,
                },
            }
            if replied_text:
                metadata["replied_message"] = {
                    "message_id": parent_id,
                    "msg_type": replied_msg_type,
                    "content": replied_text,
                }
                metadata["feishu"]["replied_message"] = metadata["replied_message"]
            if session_id and inherit_messages > 0:
                metadata["session_parent_key"] = main_session_key
                metadata["session_bootstrap_max_messages"] = inherit_messages
                metadata["feishu"]["session_parent_key"] = main_session_key
                metadata["feishu"]["session_bootstrap_max_messages"] = inherit_messages

            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_target,
                content=content,
                metadata=metadata,
                session_id=session_id,
            )
            
        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}")
