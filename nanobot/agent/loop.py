"""Agent loop: the core processing engine."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.codex_cli import CodexCLITool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _DEFERRED_REPLY_PATTERNS = [
        r"请稍等",
        r"稍等",
        r"稍后",
        r"稍后回复",
        r"稍后返回",
        r"马上返回",
        r"马上给你",
        r"马上回复",
        r"我这就",
        r"我去查",
        r"我先去查",
        r"我现在去查",
        r"一会儿",
        r"待会",
        r"\bplease wait\b",
        r"\bone moment\b",
        r"\bhang on\b",
        r"\bi(?:'| a)?m\s+(?:checking|looking up)\b",
        r"\bi(?:'ll| will)\s+(?:check|look up|get back|return)\b",
    ]
    _FORCE_FINAL_REPLY_PROMPT = (
        "Do not postpone. Either call tools now and return concrete results in this same reply, "
        "or clearly state current limits and provide the best possible direct answer now. "
        "Do not say 'please wait', 'I will return later', or similar deferred promises."
    )
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 100,
        brave_api_key: str | None = None,
        web_search_provider: str = "brave",
        searxng_base_url: str = "http://localhost:8080",
        searxng_language: str = "zh-CN",
        searxng_engines: str = "",
        web_search_fallback_to_brave: bool = True,
        web_search_timeout_seconds: float = 10.0,
        exec_config: "ExecToolConfig | None" = None,
        codex_config: "CodexToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
    ):
        from nanobot.config.schema import CodexToolConfig, ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.web_search_provider = web_search_provider
        self.searxng_base_url = searxng_base_url
        self.searxng_language = searxng_language
        self.searxng_engines = searxng_engines
        self.web_search_fallback_to_brave = web_search_fallback_to_brave
        self.web_search_timeout_seconds = web_search_timeout_seconds
        self.exec_config = exec_config or ExecToolConfig()
        self.codex_config = codex_config or CodexToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        
        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            web_search_provider=web_search_provider,
            searxng_base_url=searxng_base_url,
            searxng_language=searxng_language,
            searxng_engines=searxng_engines,
            web_search_fallback_to_brave=web_search_fallback_to_brave,
            web_search_timeout_seconds=web_search_timeout_seconds,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self._running = False
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        if self.codex_config.enabled:
            self.tools.register(CodexCLITool(
                command=self.codex_config.command,
                timeout=self.codex_config.timeout,
                default_cwd=str(self.workspace),
                default_model=self.codex_config.model,
                default_sandbox=self.codex_config.sandbox,
                default_approval=self.codex_config.approval,
                default_skip_git_repo_check=self.codex_config.skip_git_repo_check,
                max_output_chars=self.codex_config.max_output_chars,
            ))
        
        # Web tools
        self.tools.register(WebSearchTool(
            api_key=self.brave_api_key,
            provider=self.web_search_provider,
            searxng_base_url=self.searxng_base_url,
            searxng_language=self.searxng_language,
            searxng_engines=self.searxng_engines,
            fallback_to_brave=self.web_search_fallback_to_brave,
            timeout_seconds=self.web_search_timeout_seconds,
        ))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
    
    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
        
        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)
        self._bootstrap_child_session(msg, session)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)
        codex_tool = self.tools.get("codex_cli")
        if isinstance(codex_tool, CodexCLITool):
            codex_tool.set_context(msg.session_key)
        
        # Build initial messages (use get_history for LLM-formatted messages)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        
        # Agent loop
        iteration = 0
        final_content = None
        deferred_retry_count = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)  # Must be JSON string
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                
                # Execute tools
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # No tool calls: avoid deferred placeholder replies.
                content = response.content or ""
                if (
                    deferred_retry_count < 1
                    and self._looks_like_deferred_reply(content)
                ):
                    deferred_retry_count += 1
                    logger.warning(
                        "Deferred placeholder reply detected (no tools). Forcing immediate completion retry."
                    )
                    messages = self.context.add_assistant_message(
                        messages,
                        content,
                        reasoning_content=response.reasoning_content,
                    )
                    messages.append({"role": "user", "content": self._FORCE_FINAL_REPLY_PROMPT})
                    continue

                final_content = content
                break
        
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        # Log response preview
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )

    def _looks_like_deferred_reply(self, content: str) -> bool:
        """Heuristically detect replies that promise a later result without doing work now."""
        text = (content or "").strip().lower()
        if not text:
            return False
        for pattern in self._DEFERRED_REPLY_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        return False

    def _bootstrap_child_session(self, msg: InboundMessage, session: "Session") -> None:
        """Initialize a child session from parent history once, if requested by channel metadata."""
        if session.messages:
            return

        metadata = msg.metadata or {}
        parent_key = metadata.get("session_parent_key")
        if not parent_key or not isinstance(parent_key, str):
            return
        if parent_key == msg.session_key:
            return

        max_messages_raw = metadata.get("session_bootstrap_max_messages", 0)
        try:
            max_messages = int(max_messages_raw)
        except (TypeError, ValueError):
            max_messages = 0
        if max_messages <= 0:
            return

        parent = self.sessions.get_or_create(parent_key)
        if not parent.messages:
            return

        inherited = parent.messages[-max_messages:]
        session.messages = [dict(item) for item in inherited]
        session.metadata.setdefault("seeded_from_session", parent_key)
        session.metadata.setdefault("seeded_message_count", len(inherited))
        logger.info(
            "Seeded child session {} from {} with {} messages",
            msg.session_key,
            parent_key,
            len(inherited),
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(origin_channel, origin_chat_id)
        codex_tool = self.tools.get("codex_cli")
        if isinstance(codex_tool, CodexCLITool):
            codex_tool.set_context(session_key)
        
        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        
        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "Background task completed."
        
        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier.
            channel: Source channel (for context).
            chat_id: Source chat ID (for context).
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content,
            session_id=session_key,
        )
        
        response = await self._process_message(msg)
        return response.content if response else ""
