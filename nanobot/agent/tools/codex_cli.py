"""Codex CLI tool for delegated coding and analysis tasks."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


@dataclass
class _CodexRunResult:
    """Parsed result from a codex CLI run."""

    return_code: int
    message: str
    thread_id: str | None
    stderr: str


class CodexCLITool(Tool):
    """Invoke Codex CLI non-interactively, with per-session thread continuity."""

    def __init__(
        self,
        command: str = "codex",
        timeout: int = 180,
        default_cwd: str | None = None,
        default_model: str = "",
        default_sandbox: str = "workspace-write",
        default_approval: str = "never",
        default_skip_git_repo_check: bool = True,
        max_output_chars: int = 12000,
    ):
        self.command = command
        self.timeout = max(int(timeout), 1)
        self.default_cwd = default_cwd or str(Path.cwd())
        self.default_model = default_model
        self.default_sandbox = default_sandbox
        self.default_approval = default_approval
        self.default_skip_git_repo_check = default_skip_git_repo_check
        self.max_output_chars = max(int(max_output_chars), 1000)
        self._context_session_key: str | None = None
        self._thread_map_path = Path.home() / ".nanobot" / "codex_threads.json"
        self._thread_map = self._load_thread_map()

    @property
    def name(self) -> str:
        return "codex_cli"

    @property
    def description(self) -> str:
        return (
            "Delegate a coding/analysis prompt to Codex CLI (`codex exec --json`) and return its final message. "
            "Keeps session continuity automatically per chat."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Prompt sent to Codex CLI"},
                "resume": {
                    "type": "boolean",
                    "description": "Resume the stored Codex thread for this chat if available",
                    "default": True,
                },
                "thread_id": {
                    "type": "string",
                    "description": "Optional explicit Codex thread ID to resume",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for Codex CLI. Defaults to nanobot workspace.",
                },
                "model": {"type": "string", "description": "Optional Codex model override"},
                "sandbox": {
                    "type": "string",
                    "enum": ["read-only", "workspace-write", "danger-full-access"],
                    "description": "Codex sandbox mode override",
                },
                "approval": {
                    "type": "string",
                    "enum": ["untrusted", "on-request", "never"],
                    "description": "Codex approval policy override",
                },
                "skip_git_repo_check": {
                    "type": "boolean",
                    "description": "Pass --skip-git-repo-check to Codex CLI",
                    "default": True,
                },
            },
            "required": ["prompt"],
        }

    def set_context(self, session_key: str) -> None:
        """Bind this tool run context to a nanobot session key."""
        self._context_session_key = session_key

    async def execute(
        self,
        prompt: str,
        resume: bool = True,
        thread_id: str | None = None,
        cwd: str | None = None,
        model: str | None = None,
        sandbox: str | None = None,
        approval: str | None = None,
        skip_git_repo_check: bool | None = None,
        **kwargs: Any,
    ) -> str:
        prompt_text = (prompt or "").strip()
        if not prompt_text:
            return "Error: prompt is empty"

        workdir = cwd or self.default_cwd
        if not Path(workdir).exists():
            return f"Error: cwd does not exist: {workdir}"

        model_name = (model or self.default_model or "").strip()
        sandbox_mode = (sandbox or self.default_sandbox or "workspace-write").strip()
        approval_mode = (approval or self.default_approval or "never").strip()
        skip_repo_check = self.default_skip_git_repo_check if skip_git_repo_check is None else bool(skip_git_repo_check)

        stored_thread = self._get_stored_thread_id()
        target_thread = (thread_id or "").strip() or (stored_thread if resume else None)

        if target_thread:
            resumed = await self._run_codex(
                prompt=prompt_text,
                cwd=workdir,
                model=model_name,
                sandbox=sandbox_mode,
                approval=approval_mode,
                skip_git_repo_check=skip_repo_check,
                resume_thread_id=target_thread,
            )
            if resumed.return_code == 0:
                self._store_thread_id(resumed.thread_id or target_thread)
                return self._truncate_output(resumed.message)
            # If thread_id is explicitly provided by caller, don't silently fork a new thread.
            if thread_id:
                return self._format_error("codex resume failed", resumed.stderr, resumed.return_code)

        # Fallback to fresh thread when there is no stored thread or resume failed.
        fresh = await self._run_codex(
            prompt=prompt_text,
            cwd=workdir,
            model=model_name,
            sandbox=sandbox_mode,
            approval=approval_mode,
            skip_git_repo_check=skip_repo_check,
            resume_thread_id=None,
        )
        if fresh.return_code != 0:
            return self._format_error("codex exec failed", fresh.stderr, fresh.return_code)

        self._store_thread_id(fresh.thread_id)
        return self._truncate_output(fresh.message)

    def _build_base_args(
        self,
        cwd: str,
        model: str,
        sandbox: str,
        approval: str,
    ) -> list[str]:
        args = [self.command]
        if approval:
            args.extend(["-a", approval])
        if sandbox:
            args.extend(["-s", sandbox])
        if cwd:
            args.extend(["-C", cwd])
        if model:
            args.extend(["-m", model])
        return args

    async def _run_codex(
        self,
        prompt: str,
        cwd: str,
        model: str,
        sandbox: str,
        approval: str,
        skip_git_repo_check: bool,
        resume_thread_id: str | None,
    ) -> _CodexRunResult:
        cmd = self._build_base_args(
            cwd=cwd,
            model=model,
            sandbox=sandbox,
            approval=approval,
        )
        if resume_thread_id:
            cmd.extend(["exec", "resume"])
            if skip_git_repo_check:
                cmd.append("--skip-git-repo-check")
            cmd.extend(["--json", resume_thread_id, prompt])
        else:
            cmd.append("exec")
            if skip_git_repo_check:
                cmd.append("--skip-git-repo-check")
            cmd.extend(["--json", prompt])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except FileNotFoundError:
            return _CodexRunResult(
                return_code=127,
                message="",
                thread_id=None,
                stderr=f"Command not found: {self.command}",
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return _CodexRunResult(
                return_code=124,
                message="",
                thread_id=None,
                stderr=f"Timed out after {self.timeout}s",
            )

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        message, thread = self._parse_jsonl_output(stdout_text)

        return _CodexRunResult(
            return_code=process.returncode,
            message=message,
            thread_id=thread,
            stderr=stderr_text,
        )

    def _parse_jsonl_output(self, text: str) -> tuple[str, str | None]:
        latest_message = ""
        thread_id = None

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")
            if event_type == "thread.started":
                thread_id = event.get("thread_id") or thread_id
                continue

            if event_type == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message":
                    latest_message = (item.get("text") or "").strip() or latest_message

        if not latest_message:
            latest_message = "(codex returned no final message)"
        return latest_message, thread_id

    def _format_error(self, title: str, stderr: str, return_code: int) -> str:
        detail = (stderr or "").strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        if detail:
            return f"Error: {title} (exit={return_code})\n{detail}"
        return f"Error: {title} (exit={return_code})"

    def _truncate_output(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        extra = len(text) - self.max_output_chars
        return f"{text[:self.max_output_chars]}\n... (truncated, {extra} more chars)"

    def _load_thread_map(self) -> dict[str, str]:
        if not self._thread_map_path.exists():
            return {}
        try:
            data = json.loads(self._thread_map_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if k and v}

    def _save_thread_map(self) -> None:
        self._thread_map_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._thread_map_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._thread_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self._thread_map_path)

    def _get_stored_thread_id(self) -> str | None:
        if not self._context_session_key:
            return None
        return self._thread_map.get(self._context_session_key)

    def _store_thread_id(self, thread_id: str | None) -> None:
        if not thread_id or not self._context_session_key:
            return
        self._thread_map[self._context_session_key] = thread_id
        self._save_thread_map()
